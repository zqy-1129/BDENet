import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.benchmarks.bench_meijerint import alpha, bench


class SelfAttentionV1(nn.Module):
    def __init__(self, in_dim, scale=8, eps=1e-10):
        super(SelfAttentionV1, self).__init__()
        self.in_channel = in_dim
        self.scale = scale
        self.eps = eps

        # Convolutional layers for query, key, and value
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // scale, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // scale, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Additional convolutions to refine output
        self.dconv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.dconv3 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1)

        # Gamma parameter for attention scaling
        self.gamma = nn.Parameter(torch.zeros(1))

        # Softmax for attention computation
        self.softmax = nn.Softmax(dim=-1)

        # Feature map processing function (placeholder)
        self.feature_map = lambda x: x  # Replace with actual feature map if required

    def forward(self, x):
        """
        inputs :
            x : input feature map (B X C X H X W)
        returns :
            out : attention value + input feature for x
        """
        B, C, H, W = x.size()

        # Query, Key, and Value transformations for input x
        q = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)
        k = self.key_conv(x).view(B, -1, W * H)
        v = self.value_conv(x).view(B, -1, W * H)

        # Attention computation within x
        energy = torch.bmm(q, k)  # Q * K
        attention = self.softmax(energy)  # Softmax normalization
        out = torch.bmm(v, attention.permute(0, 2, 1))  # Weighted sum of values
        out = out.view(B, C, H, W)

        kv = torch.einsum("bmn, bcn->bmc", k, v)
        # Normalization (similar to ILMSPAM_Module)
        norm = 1 / (torch.einsum("bnc, bc->bn", q, torch.sum(k, dim=-1) + self.eps))
        weight_value = torch.einsum("bnm, bmc, bn->bcn", q, kv, norm)
        weight_value = weight_value.view(B, C, H, W) + self.dconv1(x) + self.dconv3(x)

        # Apply refinements using convolutions (dconv1 and dconv3)
        out = weight_value + self.dconv1(x) + self.dconv3(x)

        # Final output with attention scaling
        out = x + self.gamma * out

        return out

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

def delu_feature_map(x, param=10):
    x1 = F.relu(x)
    x2 = x - x1
    x2 = torch.exp(param * x2) - 1
    return param * x1 + x2 + 1

class LinearSelfAttention(nn.Module):
    def __init__(self, in_dim, scale=8, eps=1e-10):
        super(LinearSelfAttention, self).__init__()
        self.in_channel = in_dim
        self.scale = scale
        self.eps = eps

        # Convolutional layers for query, key, and value
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Additional convolutions for refinement
        self.dconv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.dconv3 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1)

        # Gamma parameters for attention scaling
        self.gamma = nn.Parameter(torch.zeros(1))

        # Feature map processing function (using delu_feature_map)
        self.feature_map = delu_feature_map

    def forward(self, x):
        """
        inputs :
            x : input feature map (B X C X H X W)
        returns :
            out : attention value + input feature for x
        """
        B, C, H, W = x.size()

        # Query, Key, and Value transformations for input x
        q = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key_conv(x).view(B, -1, H * W)
        v = self.value_conv(x).view(B, -1, H * W)

        # Apply feature map to Q and K
        q = self.feature_map(q).permute(0, 2, 1)  # Transpose for attention computation
        k = self.feature_map(k)

        # Ensure that the channels after query_conv and key_conv are aligned
        assert k.size(1) == v.size(1), f"Channel mismatch: {k.size(1)} != {v.size(1)}"

        # Compute energies between queries and keys using einsum
        energy = torch.einsum("bmn,bcn->bmc", q, k)  # Q * K

        # Compute KV for later use
        KV = k * v  # Element-wise multiplication along H*W

        # Normalize energies using einsum
        norm = 1 / (torch.einsum("bmc,bcn->bn", energy, k) + self.eps)

        # Weighted sum based on energies
        attention = torch.einsum("bmc,bcn,bn->bcn", energy, KV, norm)

        # Reshape the output from einsum result to (B, C, H, W)
        out = attention.view(B, C, H, W)

        # Apply refinements using convolutions (dconv1 and dconv3)
        out = out + self.dconv1(x) + self.dconv3(x)

        # Final output with attention scaling
        out = x + self.gamma * out

        return out

class DynamicAttention(nn.Module):
    def __init__(self, d_model):
        super(DynamicAttention, self).__init__()
        self.swin_dim = int(d_model * 3 / 5)
        self.conv_dim = int(d_model * 2 / 5)
        self.conv_swin = nn.Conv2d(self.swin_dim, self.swin_dim, 1)
        self.conv_resnet = nn.Conv2d(self.conv_dim, self.conv_dim, 1)

        self.attn = SelfAttentionV1(d_model)
        self.LinearAttention = LinearSelfAttention(d_model)

        self.conv_compress = nn.Conv2d(d_model, d_model // 2, 1)

    def adaptive_attention_selector(self, x):
        """
        基于输入特征的空间尺寸、通道数和稀疏性等动态选择注意力策略
        """
        seq_length = x.size(2)  # 特征的空间维度（如宽度或高度）
        num_channels = x.size(1)  # 特征的通道数

        # 计算特征的稀疏性，作为稀疏度选择的标准
        sparsity = torch.sum(x != 0, dim=-1).float() / x.size(-1)
        avg_sparsity = torch.mean(sparsity)

        # 选择注意力策略的条件
        if seq_length > 16:  # 对较长的序列使用线性注意力
            lambda_param = 0.0
        elif seq_length > 8 and avg_sparsity < 0.5:  # 对较稀疏的中等长度序列使用线性注意力
            lambda_param = 0.2
        elif seq_length > 8 and avg_sparsity >= 0.5:  # 对密集的中等长度序列使用混合策略
            lambda_param = 0.5
        else:  # 对较短的序列使用传统的softmax注意力
            lambda_param = 1.0

        # 如果通道数较大，偏向使用传统Softmax注意力
        if num_channels > 64:
            lambda_param = max(lambda_param, 0.7)

        return lambda_param

    # def adaptive_attention_selector(self, x):
    #     """
    #     基于输入特征的空间尺寸、通道数和稀疏性等动态选择注意力策略（硬选择策略）
    #     """
    #     seq_length = x.size(2)  # 特征的空间维度（如宽度或高度）
    #     num_channels = x.size(1)  # 特征的通道数
    #
    #     # 计算特征的稀疏性，作为稀疏度选择的标准
    #     sparsity = torch.sum(x != 0, dim=-1).float() / x.size(-1)
    #     avg_sparsity = torch.mean(sparsity)
    #
    #     # 基于通道数和稀疏性选择注意力策略（硬选择）
    #     if num_channels <= 160:
    #         # 较低通道数，使用线性注意力
    #         return 'linear'
    #     elif num_channels <= 640:
    #         # 中等通道数，使用混合策略（线性 + Softmax）
    #         if seq_length > 8 and avg_sparsity < 0.5:
    #             return 'linear'
    #         elif seq_length > 8 and avg_sparsity >= 0.5:
    #             return 'mixed'
    #         else:
    #             return 'softmax'
    #     else:
    #         # 高通道数，使用传统Softmax注意力
    #         return 'softmax'

    def dynamic_attention(self, x):
        lambda_param = self.adaptive_attention_selector(x)

        # if lambda_param == 'softmax':
        #     alpha = 0
        # elif lambda_param == 'mixed':
        #     alpha = 0.5
        # else:
        #     alpha = 1

        # 计算传统的softmax注意力
        softmax_output = self.attn(x)

        # 计算线性注意力
        linear_output = self.LinearAttention(x)

        # 根据lambda参数混合注意力结果
        output = lambda_param * softmax_output + (1 - lambda_param) * linear_output
        return output

    def forward(self, swin_feat, resnet_feat):
        swin_feat_aligned = self.conv_swin(swin_feat)
        resnet_feat_aligned = self.conv_resnet(resnet_feat)

        # 拼接特征
        combined_feat = torch.cat([swin_feat_aligned, resnet_feat_aligned], dim=1)

        # 动态注意力计算
        output = self.dynamic_attention(combined_feat) + combined_feat
        output = self.conv_compress(output)


        return output


if __name__ == '__main__':
    imgA = torch.randn(2, 384, 16, 16)
    imgB = torch.randn(2, 256, 16, 16)
    model = DynamicAttention(640)
    print(model(imgA, imgB).shape)
