# import torch.nn as nn
# import torch
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#
# # class AgentSpatialAttention(nn.Module):
# #     def __init__(self, in_channels, agent_num=49, window_size=(8, 8)):
# #         super(AgentSpatialAttention, self).__init__()
# #         self.in_channels = in_channels
# #         self.agent_num = agent_num
# #         self.window_size = window_size
# #
# #         # 代理特征生成器
# #         self.pool = nn.AdaptiveAvgPool2d(output_size=int(agent_num**0.5))  # 代理空间
# #         self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)  # Q, K, V生成
# #         self.softmax = nn.Softmax(dim=-1)
# #
# #         # 偏置项
# #         self.agent_bias = nn.Parameter(torch.zeros(agent_num, *window_size))
# #         trunc_normal_(self.agent_bias, std=0.02)
# #
# #         # 输出投影
# #         self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
# #
# #     def forward(self, x):
# #         b, c, h, w = x.shape
# #         # 生成代理特征
# #         agent_tokens = self.pool(x)  # (b, c, agent_h, agent_w)
# #         agent_tokens = agent_tokens.flatten(2).transpose(1, 2)  # (b, agent_num, c)
# #
# #         # Q, K, V
# #         qkv = self.qkv(x).reshape(b, 3, c, h * w).permute(1, 0, 3, 2)
# #         q, k, v = qkv[0], qkv[1], qkv[2]  # (b, h*w, c)
# #
# #         # 代理注意力计算
# #         agent_attn = self.softmax((agent_tokens @ k.transpose(-2, -1)) + self.agent_bias.flatten(1))  # (b, agent_num, h*w)
# #         agent_v = agent_attn @ v  # (b, agent_num, c)
# #
# #         # 重构到原空间
# #         x = agent_v.transpose(1, 2).view(b, c, h, w)
# #         x = self.proj(x)  # 输出投影
# #         return x
# #
# # class AgentChannelAttention(nn.Module):
# #     def __init__(self, in_channels, agent_num=64, num_heads=4):
# #         super(AgentChannelAttention, self).__init__()
# #         self.in_channels = in_channels
# #         self.agent_num = agent_num
# #         self.num_heads = num_heads
# #         head_dim = in_channels // num_heads
# #
# #         self.qkv = nn.Linear(in_channels, in_channels * 3, bias=False)
# #         self.scale = head_dim ** -0.5
# #         self.softmax = nn.Softmax(dim=-1)
# #
# #         # 代理通道生成器
# #         self.pool = nn.AdaptiveAvgPool1d(agent_num)
# #
# #         # 偏置项
# #         self.agent_bias = nn.Parameter(torch.zeros(agent_num, in_channels))
# #         trunc_normal_(self.agent_bias, std=0.02)
# #
# #         # 输出投影
# #         self.proj = nn.Linear(in_channels, in_channels)
# #
# #     def forward(self, x):
# #         b, c, h, w = x.shape
# #         x = x.flatten(2).transpose(1, 2)  # (b, hw, c)
# #
# #         # Q, K, V
# #         qkv = self.qkv(x).reshape(b, -1, 3, self.num_heads, c // self.num_heads)
# #         q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (b, hw, num_heads, head_dim)
# #
# #         # 代理通道
# #         agent_tokens = self.pool(x.transpose(1, 2))  # (b, c, agent_num)
# #         agent_tokens = agent_tokens.transpose(1, 2)  # (b, agent_num, c)
# #
# #         # 注意力计算
# #         agent_attn = self.softmax((agent_tokens @ k.transpose(-2, -1)) + self.agent_bias)  # (b, agent_num, hw)
# #         agent_v = agent_attn @ v  # (b, agent_num, head_dim)
# #
# #         # 重构通道注意力
# #         x = agent_v.transpose(1, 2).reshape(b, c, h, w)
# #         x = self.proj(x.flatten(2).transpose(1, 2)).view(b, c, h, w)  # 输出投影
# #         return x
#
# # if __name__ == '__main__':
# #     tensor = torch.randn(2, 1280, 8, 8)
# #     # agent_spatial_attention = AgentSpatialAttention(1280, 1, (1, 1))
# #     agent_channel_attention = AgentChannelAttention(1280, 1, 4)
# #     # print(agent_spatial_attention(tensor).shape)
# #     print(agent_channel_attention(tensor).shape)
#
# # class AgentPool(nn.Module):
# #     def __init__(self, pool_size):
# #         """
# #         Args:
# #             pool_size (int): 自适应池化后的尺寸 (P, P)，对应代理数为 P * P。
# #         """
# #         super().__init__()
# #         self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
# #
# #     def forward(self, x):
# #         """
# #         Args:
# #             x: 输入特征图，形状为 (B, C, H, W)。
# #         Returns:
# #             代理 token，形状为 (B, P * P, C)。
# #         """
# #         b, c, h, w = x.size()
# #         pooled = self.pool(x)  # (B, C, P, P)
# #         agent_tokens = pooled.flatten(2).transpose(1, 2)  # (B, P * P, C)
# #         return agent_tokens
# #
# # class AgentAttention(nn.Module):
# #     def __init__(self, dim, num_heads, window_size, agent_num, qkv_bias=True, attn_drop=0., proj_drop=0.):
# #         """
# #         Args:
# #             dim (int): 输入通道数。
# #             num_heads (int): 注意力头数。
# #             window_size (tuple): 窗口尺寸 (Wh, Ww)。
# #             agent_num (int): 代理 token 数量。
# #             qkv_bias (bool, optional): 是否使用偏置。默认值: True。
# #             attn_drop (float, optional): 注意力权重的 dropout 比例。默认值: 0.0。
# #             proj_drop (float, optional): 输出的 dropout 比例。默认值: 0.0。
# #         """
# #         super().__init__()
# #         self.dim = dim
# #         self.num_heads = num_heads
# #         self.agent_num = agent_num
# #         self.scale = (dim // num_heads) ** -0.5
# #
# #         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
# #         self.proj = nn.Linear(dim, dim)
# #
# #         self.attn_drop = nn.Dropout(attn_drop)
# #         self.proj_drop = nn.Dropout(proj_drop)
# #
# #         self.softmax = nn.Softmax(dim=-1)
# #         self.agent_pool = AgentPool(pool_size=int(agent_num ** 0.5))
# #
# #     def forward(self, x):
# #         """
# #         Args:
# #             x: 输入特征，形状为 (B, N, C)。
# #         Returns:
# #             输出特征，形状为 (B, N, C)。
# #         """
# #         b, n, c = x.shape
# #         qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)  # (3, B, N, C)
# #         q, k, v = qkv[0], qkv[1], qkv[2]  # 分别为 (B, N, C)
# #
# #         # 代理 token 计算
# #         agent_tokens = self.agent_pool(q.reshape(b, int(n**0.5), int(n**0.5), c).permute(0, 3, 1, 2))
# #
# #         # Multi-head Self-Attention with agents
# #         q = q.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
# #         k = k.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
# #         v = v.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
# #
# #         agent_tokens = agent_tokens.reshape(b, self.agent_num, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
# #
# #         attn = self.softmax((q @ k.transpose(-2, -1)) * self.scale)  # (B, num_heads, N, N)
# #         attn = self.attn_drop(attn)
# #
# #         x = (attn @ v).transpose(1, 2).reshape(b, n, c)
# #
# #         # 代理特征融合
# #         agent_attn = self.softmax((agent_tokens @ k.transpose(-2, -1)) * self.scale)
# #         agent_attn = self.attn_drop(agent_attn)
# #         agent_out = agent_attn @ v
# #
# #         # 合并输出
# #         x = x + agent_out.transpose(1, 2).reshape(b, n, c)
# #
# #         x = self.proj(x)
# #         x = self.proj_drop(x)
# #
# #         return x
# #
# # class MultiStageAttention(nn.Module):
# #     def __init__(self, dims, num_heads, window_sizes, agent_nums):
# #         """
# #         Args:
# #             dims (list): 每个阶段的通道数。
# #             num_heads (list): 每个阶段的注意力头数。
# #             window_sizes (list): 每个阶段的窗口大小。
# #             agent_nums (list): 每个阶段的代理 token 数。
# #         """
# #         super().__init__()
# #         self.attentions = nn.ModuleList([
# #             AgentAttention(dim=dims[i], num_heads=num_heads[i], window_size=window_sizes[i], agent_num=agent_nums[i])
# #             for i in range(len(dims))
# #         ])
# #
# #     def forward(self, features):
# #         """
# #         Args:
# #             features: 输入的多阶段特征列表，每个元素形状为 (B, C, H, W)。
# #         Returns:
# #             输出的多阶段特征列表，每个元素形状为 (B, C, H, W)。
# #         """
# #         outputs = []
# #         for i, feature in enumerate(features):
# #             b, c, h, w = feature.size()
# #             feature_flat = feature.flatten(2).transpose(1, 2)  # (B, H*W, C)
# #             out = self.attentions[i](feature_flat)
# #             outputs.append(out.transpose(1, 2).reshape(b, c, h, w))
# #         return outputs
#
# # 示例
# # if __name__ == "__main__":
# #     features = [
# #         torch.rand(8, 160, 64, 64),
# #         torch.rand(8, 320, 32, 32),
# #         torch.rand(8, 640, 16, 16),
# #         torch.rand(8, 1280, 8, 8)
# #     ]
# #
# #     dims = [160, 320, 640, 1280]
# #     num_heads = [4, 8, 16, 32]
# #     window_sizes = [(8, 8), (4, 4), (2, 2), (1, 1)]
# #     agent_nums = [64, 16, 4, 1]
# #
# #     model = MultiStageAttention(dims, num_heads, window_sizes, agent_nums)
# #     outputs = model(features)
# #
# #     for i, out in enumerate(outputs):
# #         print(f"Stage {i+1} output shape: {out.shape}")
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class AgentPool(nn.Module):
#     def __init__(self, pool_size):
#         """
#         Args:
#             pool_size (int): 自适应池化后的尺寸 (P, P)，对应代理数为 P * P。
#         """
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
#
#     def forward(self, x):
#         """
#         Args:
#             x: 输入特征图，形状为 (B, C, H, W)。
#         Returns:
#             代理 token，形状为 (B, P * P, C)。
#         """
#         b, c, h, w = x.size()
#         pooled = self.pool(x)  # (B, C, P, P)
#         agent_tokens = pooled.flatten(2).transpose(1, 2)  # (B, P * P, C)
#         return agent_tokens
#
# class AgentAttention(nn.Module):
#     # 原有的 __init__ 方法保持不变
#     def __init__(self, dim, num_heads, window_size, agent_num, qkv_bias=True, attn_drop=0., proj_drop=0.):
#         """
#         Args:
#             dim (int): 输入通道数。
#             num_heads (int): 注意力头数。
#             window_size (tuple): 窗口尺寸 (Wh, Ww)。
#             agent_num (int): 代理 token 数量。
#             qkv_bias (bool, optional): 是否使用偏置。默认值: True。
#             attn_drop (float, optional): 注意力权重的 dropout 比例。默认值: 0.0。
#             proj_drop (float, optional): 输出的 dropout 比例。默认值: 0.0。
#         """
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.agent_num = agent_num
#         self.scale = (dim // num_heads) ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.proj = nn.Linear(dim, dim)
#
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.softmax = nn.Softmax(dim=-1)
#         self.agent_pool = AgentPool(pool_size=int(agent_num ** 0.5))
#     def forward(self, x):
#         b, n, c = x.shape
#
#         # 获取 query, key, value
#         qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # (B, N, C)
#
#         # 代理 token 计算
#         agent_tokens = self.agent_pool(
#             q.reshape(b, int(n**0.5), int(n**0.5), c).permute(0, 3, 1, 2)
#         )  # 形状 (B, agent_num, C)
#
#         # Multi-head Self-Attention
#         q = q.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
#         k = k.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
#         v = v.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
#
#         # 代理特征处理
#         agent_tokens = agent_tokens.reshape(
#             b, self.agent_num, self.num_heads, c // self.num_heads
#         ).permute(0, 2, 1, 3)
#
#         # Self-attention 权重计算
#         attn = self.softmax((q @ k.transpose(-2, -1)) * self.scale)  # (B, num_heads, N, N)
#         attn = self.attn_drop(attn)
#
#         # Self-attention 输出
#         # x = (attn @ v).transpose(1, 2).reshape(b, n, c)
#
#         # 代理注意力
#         agent_attn = self.softmax((agent_tokens @ k.transpose(-2, -1)) * self.scale)
#         agent_attn = self.attn_drop(agent_attn)
#         agent_out = agent_attn @ v
#
#         # 合并输出
#         agent_out = agent_out.mean(dim=1)  # (B, N, C)，使用均值确保形状匹配
#         # agent_out = agent_out.permute(0, 2, 1).reshape(b, c, n)  # (B, C, N)
#         # agent_out = agent_out.transpose(1, 2)  # (B, N, C)
#         x = x + agent_out
#
#         # 最终投影
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         return x
#
#
# class ChannelAttention(nn.Module):
#     def __init__(self, dim, reduction=16):
#         """
#         Args:
#             dim (int): 输入通道数。
#             reduction (int): 通道降维倍率，默认值为 16。
#         """
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(dim, dim // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(dim // reduction, dim, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         """
#         Args:
#             x: 输入特征，形状为 (B, C, H, W)。
#         Returns:
#             输出特征，形状为 (B, C, H, W)。
#         """
#         b, c, h, w = x.size()
#         y = self.avg_pool(x).view(b, c)  # (B, C)
#         y = self.fc(y).view(b, c, 1, 1)  # (B, C, 1, 1)
#         return x * y.expand_as(x)
#
# class MultiStageAttention(nn.Module):
#     def __init__(self, dims, num_heads, window_sizes, agent_nums):
#         """
#         Args:
#             dims (list): 每个阶段的通道数。
#             num_heads (list): 每个阶段的注意力头数。
#             window_sizes (list): 每个阶段的窗口大小。
#             agent_nums (list): 每个阶段的代理 token 数。
#         """
#         super().__init__()
#         self.attentions = nn.ModuleList([
#             AgentAttention(dim=dims[i], num_heads=num_heads[i], window_size=window_sizes[i], agent_num=agent_nums[i])
#             for i in range(len(dims))
#         ])
#         self.channel_attentions = nn.ModuleList([
#             ChannelAttention(dim=dims[i]) for i in range(len(dims))
#         ])
#
#     def forward(self, features):
#         """
#         Args:
#             features: 输入的多阶段特征列表，每个元素形状为 (B, C, H, W)。
#         Returns:
#             输出的多阶段特征列表，每个元素形状为 (B, C, H, W)。
#         """
#         outputs = []
#         for i, feature in enumerate(features):
#             b, c, h, w = feature.size()
#             feature_flat = feature.flatten(2).transpose(1, 2)  # (B, H*W, C)
#             agent_out = self.attentions[i](feature_flat)
#             agent_out = agent_out.transpose(1, 2).reshape(b, c, h, w)
#             channel_out = self.channel_attentions[i](agent_out)
#             outputs.append(channel_out)
#         return outputs
#
# # 示例
# if __name__ == "__main__":
#     features = [
#         torch.rand(8, 160, 64, 64),
#         torch.rand(8, 320, 32, 32),
#         torch.rand(8, 640, 16, 16),
#         torch.rand(8, 1280, 8, 8)
#     ]
#
#     dims = [160, 320, 640, 1280]
#     num_heads = [1, 1, 1, 1]
#     window_sizes = [(8, 8), (4, 4), (2, 2), (1, 1)]
#     agent_nums = [64, 16, 4, 1]
#
#     model = MultiStageAttention(dims, num_heads, window_sizes, agent_nums)
#     outputs = model(features)
#
#     for i, out in enumerate(outputs):
#         print(f"Stage {i+1} output shape: {out.shape}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_

class ProxySpatialAttentionModule(nn.Module):
    def __init__(self, in_dim, num_heads=8, window_size=7):
        super(ProxySpatialAttentionModule, self).__init__()
        self.in_channel = in_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = in_dim // num_heads
        assert self.head_dim * num_heads == in_dim, "in_dim must be divisible by num_heads"

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, groups=num_heads)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, groups=num_heads)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, groups=num_heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        # Transformations
        query = self.query_conv(x).view(B * self.num_heads, self.head_dim, H * W)
        key = self.key_conv(x).view(B * self.num_heads, self.head_dim, H * W)
        value = self.value_conv(x).view(B * self.num_heads, self.head_dim, H * W)

        # Compute attention
        energy = torch.bmm(query.transpose(1, 2), key)  # (B * num_heads) x (H * W) x (H * W)
        attention = self.softmax(energy)
        out = torch.bmm(value, attention).view(B, C, H, W)

        # Combine with input feature
        out = out + x
        return out

class ProxyChannelAttentionModule(nn.Module):
    def __init__(self, in_dim, num_heads=8):
        super(ProxyChannelAttentionModule, self).__init__()
        self.in_channel = in_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        assert self.head_dim * num_heads == in_dim, "in_dim must be divisible by num_heads"

        self.query_conv = nn.Linear(in_dim, in_dim, bias=False)
        self.key_conv = nn.Linear(in_dim, in_dim, bias=False)
        self.value_conv = nn.Linear(in_dim, in_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        x_flat = x.view(B, C, -1).transpose(1, 2).contiguous()  # B x (H*W) x C
        x_flat = x_flat.view(-1, C)  # (B*H*W) x C

        # Transformations
        query = self.query_conv(x_flat).view(B * H * W * self.num_heads, self.head_dim)
        key = self.key_conv(x_flat).view(B * H * W * self.num_heads, self.head_dim)
        value = self.value_conv(x_flat).view(B * H * W * self.num_heads, self.head_dim)

        # Compute attention
        energy = torch.bmm(query.unsqueeze(1), key.unsqueeze(2))  # (B*H*W*num_heads) x 1 x 1
        energy = energy.squeeze(-1).squeeze(-1)  # (B*H*W*num_heads), remove last two singleton dimensions

        attention = self.softmax(energy).unsqueeze(1)  # (B*H*W*num_heads) x 1 x 1

        out = torch.bmm(attention, value.unsqueeze(2)).squeeze(2)  # (B*H*W*num_heads) x head_dim
        out = out.view(B, H * W, self.head_dim * self.num_heads).transpose(1, 2).contiguous()  # B x C x (H*W)
        out = out.view(B, C, H, W)

        # Combine with input feature
        out = out + x
        return out

class AgentAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 agent_num=49, window=14, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num
        self.window = window

        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
                             padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
        self.ac_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1))
        self.ca_bias = nn.Parameter(torch.zeros(1, num_heads, 1, agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        trunc_normal_(self.ac_bias, std=.02)
        trunc_normal_(self.ca_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q, k, v: b, n, c

        agent_tokens = self.pool(q[:, :, :].reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        # position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
        # position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        # position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        # position_bias = position_bias1 + position_bias2
        # position_bias = torch.cat([self.ac_bias.repeat(b, 1, 1, 1), position_bias], dim=-1)
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1))
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        # agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
        # agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        # agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        # agent_bias = agent_bias1 + agent_bias2
        # agent_bias = torch.cat([self.ca_bias.repeat(b, 1, 1, 1), agent_bias], dim=-2)
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) )#+ agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v_ = v[:, :, :, :].transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x[:, :, :] = x[:, :, :] + self.dwc(v_).permute(0, 2, 3, 1).reshape(b, n , c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

if __name__ == '__main__':
    # x = torch.randn(2, 160, 64, 64)
    # model = ProxySpatialAttentionModule(160, num_heads=1, window_size=8)
    # # model_channel = ProxyChannelAttentionModule(160, num_heads=1)
    # print(model(x).shape)
    y = torch.randn(2, 64 * 64, 64)
    model_channel = AgentAttention(64, num_heads=1, window=8)
    print(model_channel(y).shape)
    # print(model_channel(x).shape)