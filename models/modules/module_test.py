import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(DeformConv2D, self).__init__()
        self.conv_offset = nn.Conv2d(in_channels, 2 * kernel_size ** 2, kernel_size=3, padding=1, stride=stride)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        offset = self.conv_offset(x)
        # Reshape to (batch_size, 2, height, width)
        batch_size = offset.data.size()[0]
        x = x.unsqueeze(1)
        # B, 1, H, W -> B, H, W, 1
        offset = offset.view(batch_size, 2, -1, offset.size(3))
        # B, H, W, 1 -> B, H, W, 1
        offset_h = offset[:, 0, :, :]
        # B, H, W, 1 -> B, H, W, 1
        offset_w = offset[:, 1, :, :]
        # Get the actual coordinates by adding the offsets to the initial pixel locations
        x_coord, y_coord = torch.arange(x.size(2), dtype=torch.float, device=x.device) + offset_h, torch.arange(x.size(3), dtype=torch.float, device=x.device).unsqueeze(0) + offset_w
        # Flip coordinates if stride is negative
        if self.conv.stride[0] < 0:
            x_coord = x_coord[ :, torch.arange(x_coord.size(1) - 1, -1, -1), :]
        if self.conv.stride[1] < 0:
            y_coord = y_coord[:, :, torch.arange(y_coord.size(2) - 1, -1, -1)]
        # Perform normal convolution using the adjusted coordinates
        x_coord = x_coord.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        y_coord = y_coord.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        output = F.conv2d(x, self.conv.weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)
        output = output.view(batch_size, output.size(1), x.size(2), x.size(3))
        return output

# 使用示例
# model = DeformConv2D(16, 32, kernel_size=3, stride=1, padding=1)
# input = torch.randn(1, 16, 100, 100)
# output = model(input)
# print(output.shape)

import timm

# List all available models
model_names = timm.list_models()
for model_name in model_names:
    print(model_name)