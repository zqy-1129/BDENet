import torch
import torch.nn as nn
import torch.nn.functional as F
# 改进后的交错注意力。非线性注意力，上个版本的LinearAttention
class CrossAttentionV1(nn.Module):
    def __init__(self, in_dim, scale=8, eps=1e-10):
        super(CrossAttentionV1, self).__init__()
        self.in_channel = in_dim
        self.scale = scale
        self.eps = eps

        # Convolutional layers for query, key, and value
        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // scale, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // scale, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.query_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // scale, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // scale, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Additional convolutions to refine output
        self.dconv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.dconv3 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1)

        # Gamma parameters for attention scaling
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        # Softmax for attention computation
        self.softmax = nn.Softmax(dim=-1)

        # Feature map processing function (placeholder)
        self.feature_map = lambda x: x  # Replace with actual feature map if required

    def forward(self, x1, x2):
        """
        inputs :
            x1 : input feature map 1 (B X C X H X W)
            x2 : input feature map 2 (B X C X H X W)
        returns :
            out1 : attention value + input feature for x1
            out2 : attention value + input feature for x2
            attention: (B X (HxW) X (HxW)) (optional)
        """
        B, C, H, W = x1.size()

        # Query, Key, and Value transformations for both inputs
        q1 = self.query_conv1(x1).view(B, -1, W * H).permute(0, 2, 1)
        k1 = self.key_conv1(x1).view(B, -1, W * H)
        v1 = self.value_conv1(x1).view(B, -1, W * H)

        q2 = self.query_conv2(x2).view(B, -1, W * H).permute(0, 2, 1)
        k2 = self.key_conv2(x2).view(B, -1, W * H)
        v2 = self.value_conv2(x2).view(B, -1, W * H)

        # Attention computation between x1 and x2
        energy1 = torch.bmm(q1, k2)
        attention1 = self.softmax(energy1)
        out1 = torch.bmm(v2, attention1.permute(0, 2, 1))
        out1 = out1.view(B, C, H, W)

        energy2 = torch.bmm(q2, k1)
        attention2 = self.softmax(energy2)
        out2 = torch.bmm(v1, attention2.permute(0, 2, 1))
        out2 = out2.view(B, C, H, W)

        # Normalization (similar to ILMSPAM_Module)
        norm1 = 1 / torch.einsum("bnc, bc->bn", q1, torch.sum(k1, dim=-1) + self.eps)
        norm2 = 1 / torch.einsum("bnc, bc->bn", q2, torch.sum(k2, dim=-1) + self.eps)

        # Weighting the outputs and applying refinement convolutions
        out1 = out1 + self.dconv1(x1) + self.dconv3(x1)
        out2 = out2 + self.dconv1(x2) + self.dconv3(x2)

        # Final output with attention scaling
        out1 = x1 + self.gamma1 * out1
        out2 = x2 + self.gamma2 * out2

        return out1, out2

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

def delu_feature_map(x, param=10):
    x1 = F.relu(x)
    x2 = x - x1
    x2 = torch.exp(param * x2) - 1
    return param * x1 + x2 + 1

# 改进后的交错注意力。线性注意力，当前版本的LinearAttention
class LinearCrossAttention(nn.Module):
    def __init__(self, in_dim, scale=8, eps=1e-10):
        super(LinearCrossAttention, self).__init__()
        self.in_channel = in_dim
        self.scale = scale
        self.eps = eps

        # Convolutional layers for query, key, and value
        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.query_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Additional convolutions for refinement
        self.dconv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.dconv3 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1)

        # Gamma parameters for attention scaling
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        # Feature map processing function (using delu_feature_map)
        self.feature_map = delu_feature_map

    def forward(self, x1, x2):
        """
        inputs :
            x1 : input feature map 1 (B X C X H X W)
            x2 : input feature map 2 (B X C X H X W)
        returns :
            out1 : attention value + input feature for x1
            out2 : attention value + input feature for x2
        """
        B, C, H, W = x1.size()

        # Query, Key, and Value transformations for both inputs
        q1 = self.query_conv1(x1).view(B, -1, H * W).permute(0, 2, 1)
        k1 = self.key_conv1(x1).view(B, -1, H * W)
        v1 = self.value_conv1(x1).view(B, -1, H * W)

        q2 = self.query_conv2(x2).view(B, -1, H * W).permute(0, 2, 1)
        k2 = self.key_conv2(x2).view(B, -1, H * W)
        v2 = self.value_conv2(x2).view(B, -1, H * W)

        # Apply feature map to Q and K
        q1 = self.feature_map(q1).permute(0, 2, 1)  # Transpose for attention computation
        k1 = self.feature_map(k1)

        q2 = self.feature_map(q2).permute(0, 2, 1)
        k2 = self.feature_map(k2)

        # Ensure that the channels after query_conv and key_conv are aligned
        assert k1.size(1) == v1.size(1), f"Channel mismatch: {k1.size(1)} != {v1.size(1)}"
        assert k2.size(1) == v2.size(1), f"Channel mismatch: {k2.size(1)} != {v2.size(1)}"

        # Compute energies between queries and keys using einsum
        energy1 = torch.einsum("bmn,bcn->bmc", q1, k2)  # Q1 * K2
        energy2 = torch.einsum("bmn,bcn->bmc", q2, k1)  # Q2 * K1

        # Compute KV for later use
        KV1 = k1 * v1  # Element-wise multiplication along H*W
        KV2 = k2 * v2

        # Normalize energies using einsum
        norm1 = 1 / (torch.einsum("bmc,bcn->bn", energy1, k2) + self.eps)
        norm2 = 1 / (torch.einsum("bmc,bcn->bn", energy2, k1) + self.eps)

        # Weighted sum based on energies
        attention1 = torch.einsum("bmc,bcn,bn->bcn", energy1, KV1, norm1)
        attention2 = torch.einsum("bmc,bcn,bn->bcn", energy2, KV2, norm2)

        # Reshape the output from einsum result to (B, C, H, W)
        out1 = attention1.view(B, C, H, W)
        out2 = attention2.view(B, C, H, W)

        # Apply refinements using convolutions (dconv1 and dconv3)
        out1 = out1 + self.dconv1(x1) + self.dconv3(x1)
        out2 = out2 + self.dconv1(x2) + self.dconv3(x2)

        # Final output with attention scaling
        out1 = x1 + self.gamma1 * out1
        out2 = x2 + self.gamma2 * out2

        return out1, out2



if __name__ == '__main__':
    imgA = torch.randn(2, 32, 32, 32)
    imgB = torch.randn(2, 32, 32, 32)
    model = LinearCrossAttention(32)
    print(model(imgA, imgB)[0].shape)
