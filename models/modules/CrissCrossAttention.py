import torch
import torch.nn as nn
from torch.nn import Softmax


# v1
# class CrissCrossAttention(nn.Module):
#     def __init__(self, in_dim):
#         super(CrissCrossAttention, self).__init__()
#         self.query_conv_A = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.key_conv_A = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.value_conv_A = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#
#         self.query_conv_B = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.key_conv_B = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.value_conv_B = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.softmax = Softmax(dim=3)
#         self.INF = INF
#         self.gamma_A = nn.Parameter(torch.zeros(1))
#         self.gamma_B = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x1, x2):
#         """
#         inputs :
#                 x1 : input feature maps( B X C X H X W)
#                 x2 : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X (HxW) X (HxW)
#         """
#
#         m_batchsize, _, height, width = x1.size()
#         proj_query_B = self.query_conv_A(x2)
#         proj_query_B_H = proj_query_B.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
#         proj_query_B_W = proj_query_B.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
#         proj_key_A = self.key_conv_A(x1)
#         proj_key_A_H = proj_key_A.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
#         proj_key_A_W = proj_key_A.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
#         proj_value_A = self.value_conv_A(x1)
#         proj_value_A_H = proj_value_A.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
#         proj_value_A_W = proj_value_A.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
#         energy_A_H = (torch.bmm(proj_query_B_H, proj_key_A_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
#         energy_A_W = torch.bmm(proj_query_B_W, proj_key_A_W).view(m_batchsize,height,width,width)
#         concate_A = self.softmax(torch.cat([energy_A_H, energy_A_W], 3))
#
#         att_A_H = concate_A[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
#         att_A_W = concate_A[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
#         out_A_H = torch.bmm(proj_value_A_H, att_A_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
#         out_A_W = torch.bmm(proj_value_A_W, att_A_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
#         #print(out_H.size(),out_W.size())
#
#         proj_query_A = self.query_conv_B(x1)
#         proj_query_A_H = proj_query_A.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
#         proj_query_A_W = proj_query_A.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
#         proj_key_B = self.key_conv_B(x1)
#         proj_key_B_H = proj_key_B.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
#         proj_key_B_W = proj_key_B.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
#         proj_value_B = self.value_conv_B(x1)
#         proj_value_B_H = proj_value_B.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
#         proj_value_B_W = proj_value_B.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
#         energy_B_H = (torch.bmm(proj_query_A_H, proj_key_B_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
#         energy_B_W = torch.bmm(proj_query_A_W, proj_key_B_W).view(m_batchsize,height,width,width)
#         concate_B = self.softmax(torch.cat([energy_B_H, energy_B_W], 3))
#
#         att_B_H = concate_B[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
#         att_B_W = concate_B[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
#         out_B_H = torch.bmm(proj_value_B_H, att_B_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
#         out_B_W = torch.bmm(proj_value_B_W, att_B_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
#         #print(out_H.size(),out_W.size())
#
#
#         return self.gamma_A*(out_A_H + out_A_W) + x1, self.gamma_B*(out_B_H + out_B_W) + x2

# v3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # 导入图卷积模块


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)
# V2
class CrissCrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        # A时相和B时相的卷积层
        self.query_conv_A = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv_A = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv_A = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.query_conv_B = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv_B = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv_B = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Softmax函数
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma_A = nn.Parameter(torch.zeros(1))
        self.gamma_B = nn.Parameter(torch.zeros(1))

        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(in_dim)
        )

        # 创建图卷积层
        # self.gcn = GCNConv(in_dim, in_dim)


    def forward(self, x1, x2):
        """
        inputs :
            x1 : input feature maps( B X C X H X W) for A phase
            x2 : input feature maps( B X C X H X W) for B phase
        returns :
            out : attention value + input feature
            attention: B X (HxW) X (HxW)
        """

        m_batchsize, _, height, width = x1.size()

        # A和B之间的交互部分
        # 1. 计算A时相特征对B时相的影响
        proj_query_B = self.query_conv_A(x2)
        proj_query_B_H = proj_query_B.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0,
                                                                                                                     2,
                                                                                                                     1)
        proj_query_B_W = proj_query_B.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0,
                                                                                                                     2,
                                                                                                                     1)

        proj_key_A = self.key_conv_A(x1)
        proj_key_A_H = proj_key_A.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_A_W = proj_key_A.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value_A = self.value_conv_A(x1)
        proj_value_A_H = proj_value_A.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_A_W = proj_value_A.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        energy_A_H = (torch.bmm(proj_query_B_H, proj_key_A_H) + self.INF(m_batchsize, height, width)).view(m_batchsize,
                                                                                                           width,
                                                                                                           height,
                                                                                                           height).permute(
            0, 2, 1, 3)
        energy_A_W = torch.bmm(proj_query_B_W, proj_key_A_W).view(m_batchsize, height, width, width)

        concate_A = self.softmax(torch.cat([energy_A_H, energy_A_W], 3))

        att_A_H = concate_A[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height,
                                                                                     height)
        att_A_W = concate_A[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)

        # 2. 计算B时相特征对A时相的影响
        out_A_H = torch.bmm(proj_value_A_H, att_A_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2,
                                                                                                                   3, 1)
        out_A_W = torch.bmm(proj_value_A_W, att_A_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2,
                                                                                                                   1, 3)

        # 3. 计算B时相特征对A时相的影响
        proj_query_A = self.query_conv_B(x1)
        proj_query_A_H = proj_query_A.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0,
                                                                                                                     2,
                                                                                                                     1)
        proj_query_A_W = proj_query_A.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0,
                                                                                                                     2,
                                                                                                                     1)

        proj_key_B = self.key_conv_B(x1)
        proj_key_B_H = proj_key_B.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_B_W = proj_key_B.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value_B = self.value_conv_B(x1)
        proj_value_B_H = proj_value_B.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_B_W = proj_value_B.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        energy_B_H = (torch.bmm(proj_query_A_H, proj_key_B_H) + self.INF(m_batchsize, height, width)).view(m_batchsize,
                                                                                                           width,
                                                                                                           height,
                                                                                                           height).permute(
            0, 2, 1, 3)
        energy_B_W = torch.bmm(proj_query_A_W, proj_key_B_W).view(m_batchsize, height, width, width)

        concate_B = self.softmax(torch.cat([energy_B_H, energy_B_W], 3))

        att_B_H = concate_B[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height,
                                                                                     height)
        att_B_W = concate_B[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)

        # 4. 输出部分
        out_B_H = torch.bmm(proj_value_B_H, att_B_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2,
                                                                                                                   3, 1)
        out_B_W = torch.bmm(proj_value_B_W, att_B_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2,
                                                                                                                   1, 3)
        out_A = out_A_H + out_A_W  # A
        out_B = out_B_H + out_B_W  # B

        difference = torch.abs(out_A - out_B)
        out = self.proj(difference)

        # 最终输出
        return self.gamma_A * out_A + x1 + out, self.gamma_B * out_B + x2 + out


# class CrissCrossAttention(nn.Module):
#     def __init__(self, in_dim):
#         super(CrissCrossAttention, self).__init__()
#         # A时相和B时相的卷积层
#         self.query_conv_A = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.key_conv_A = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.value_conv_A = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#
#         self.query_conv_B = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.key_conv_B = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.value_conv_B = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#
#         # Softmax函数
#         self.softmax = nn.Softmax(dim=3)
#         self.INF = lambda *size: torch.full(size, float('-inf')).to(
#             torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#         self.gamma_A = nn.Parameter(torch.zeros(1))
#         self.gamma_B = nn.Parameter(torch.zeros(1))
#
#         # 图卷积层
#         self.gcn_A = GCNConv(in_dim, in_dim)  # 图卷积层A
#         self.gcn_B = GCNConv(in_dim, in_dim)  # 图卷积层B
#
#     def forward(self, x1, x2):
#         """
#         inputs :
#             x1 : input feature maps( B X C X H X W) for A phase
#             x2 : input feature maps( B X C X H X W) for B phase
#         returns :
#             out : attention value + input feature
#             attention: B X (HxW) X (HxW)
#         """
#
#         m_batchsize, _, height, width = x1.size()
#
#         # A和B之间的交互部分
#         # 1. 计算A时相特征对B时相的影响
#         proj_query_B = self.query_conv_A(x2)
#         proj_query_B_H = proj_query_B.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0,
#                                                                                                                      2,
#                                                                                                                      1)
#         proj_query_B_W = proj_query_B.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0,
#                                                                                                                      2,
#                                                                                                                      1)
#
#         proj_key_A = self.key_conv_A(x1)
#         proj_key_A_H = proj_key_A.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
#         proj_key_A_W = proj_key_A.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
#
#         proj_value_A = self.value_conv_A(x1)
#         proj_value_A_H = proj_value_A.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
#         proj_value_A_W = proj_value_A.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
#
#         energy_A_H = (torch.bmm(proj_query_B_H, proj_key_A_H) + self.INF(m_batchsize, height, width)).view(m_batchsize,
#                                                                                                            width,
#                                                                                                            height,
#                                                                                                            height).permute(
#             0, 2, 1, 3)
#         energy_A_W = torch.bmm(proj_query_B_W, proj_key_A_W).view(m_batchsize, height, width, width)
#
#         concate_A = self.softmax(torch.cat([energy_A_H, energy_A_W], 3))
#
#         att_A_H = concate_A[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height,
#                                                                                      height)
#         att_A_W = concate_A[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
#
#         # 2. 计算B时相特征对A时相的影响
#         out_A_H = torch.bmm(proj_value_A_H, att_A_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2,
#                                                                                                                    3, 1)
#         out_A_W = torch.bmm(proj_value_A_W, att_A_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2,
#                                                                                                                    1, 3)
#
#         # 3. 计算B时相特征对A时相的影响
#         proj_query_A = self.query_conv_B(x1)
#         proj_query_A_H = proj_query_A.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0,
#                                                                                                                      2,
#                                                                                                                      1)
#         proj_query_A_W = proj_query_A.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0,
#                                                                                                                      2,
#                                                                                                                      1)
#
#         proj_key_B = self.key_conv_B(x1)
#         proj_key_B_H = proj_key_B.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
#         proj_key_B_W = proj_key_B.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
#
#         proj_value_B = self.value_conv_B(x1)
#         proj_value_B_H = proj_value_B.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
#         proj_value_B_W = proj_value_B.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
#
#         energy_B_H = (torch.bmm(proj_query_A_H, proj_key_B_H) + self.INF(m_batchsize, height, width)).view(m_batchsize,
#                                                                                                            width,
#                                                                                                            height,
#                                                                                                            height).permute(
#             0, 2, 1, 3)
#         energy_B_W = torch.bmm(proj_query_A_W, proj_key_B_W).view(m_batchsize, height, width, width)
#
#         concate_B = self.softmax(torch.cat([energy_B_H, energy_B_W], 3))
#
#         att_B_H = concate_B[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height,
#                                                                                      height)
#         att_B_W = concate_B[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
#
#         # 4. 输出部分
#         out_B_H = torch.bmm(proj_value_B_H, att_B_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2,
#                                                                                                                    3, 1)
#         out_B_W = torch.bmm(proj_value_B_W, att_B_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2,
#                                                                                                                    1, 3)
#
#         # 使用图卷积对特征进行交互
#         out_A_H = self.gcn_A(out_A_H.view(m_batchsize, -1, height * width).permute(0, 2, 1)).view(m_batchsize, -1,
#                                                                                                   height, width)
#         out_A_W = self.gcn_A(out_A_W.view(m_batchsize, -1, height * width).permute(0, 2, 1)).view(m_batchsize, -1,
#                                                                                                   height, width)
#
#         out_B_H = self.gcn_B(out_B_H.view(m_batchsize, -1, height * width).permute(0, 2, 1)).view(m_batchsize, -1,
#                                                                                                   height, width)
#         out_B_W = self.gcn_B(out_B_W.view(m_batchsize, -1, height * width).permute(0, 2, 1)).view(m_batchsize, -1,
#                                                                                                   height, width)
#
#         # 最终输出
#         return self.gamma_A * (out_A_H + out_A_W) + x1, self.gamma_B * (out_B_H + out_B_W) + x2


if __name__ == '__main__':
    feature_A = torch.randn(2, 32, 64, 64).to('cuda')
    feature_B = torch.randn(2, 32, 64, 64).to('cuda')
    model = CrissCrossAttention(32).to('cuda')
    print(model(feature_A, feature_B)[0].shape)