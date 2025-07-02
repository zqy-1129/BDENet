import torch
import torch.nn as nn
import torch.nn.functional as F
from pyexpat import features

# V1
class SceneRelation(nn.Module):
    def __init__(self,
                 in_channels,
                 channel_list,
                 out_channels,
                 scale_aware_proj=True):
        super(SceneRelation, self).__init__()
        self.scale_aware_proj = scale_aware_proj

        if scale_aware_proj:
            self.scene_encoder = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, 1),
                ) for _ in range(len(channel_list))]
            )
        else:
            # 2mlp
            self.scene_encoder = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 1),
            )
        self.content_encoders = nn.ModuleList()
        self.feature_reencoders = nn.ModuleList()
        for c in channel_list:
            self.content_encoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )
            self.feature_reencoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )

        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature, features: list):
        content_feats = [c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)]
        if self.scale_aware_proj:
            scene_feats = [op(scene_feature) for op in self.scene_encoder]
            relations = [self.normalizer((sf * cf).sum(dim=1, keepdim=True)) for sf, cf in
                         zip(scene_feats, content_feats)]
        else:
            scene_feat = self.scene_encoder(scene_feature)
            relations = [self.normalizer((scene_feat * cf).sum(dim=1, keepdim=True)) for cf in content_feats]

        p_feats = [op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)]

        refined_feats = [r * p for r, p in zip(relations, p_feats)]

        return refined_feats


# V2
# class FSRelation(nn.Module):
#     def __init__(self,
#                  scene_embedding_channels,
#                  in_channels_list,
#                  out_channels,
#                  scale_aware_proj=False,
#                  ):
#         super(FSRelation, self).__init__()
#         self.scale_aware_proj = scale_aware_proj
#
#         if scale_aware_proj:
#             self.scene_encoder = nn.ModuleList(
#                 [nn.Sequential(
#                     nn.Conv2d(scene_embedding_channels, out_channels, 1),
#                     nn.GroupNorm(32, out_channels),
#                     nn.ReLU(True),
#                     nn.Conv2d(out_channels, out_channels, 1),
#                     nn.GroupNorm(32, out_channels),
#                     nn.ReLU(True),
#                 ) for _ in range(len(in_channels_list))]
#             )
#             self.project = nn.ModuleList(
#                 [nn.Sequential(
#                     nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
#                     nn.BatchNorm2d(out_channels),
#                     nn.ReLU(True),
#                     nn.Dropout2d(p=0.1)
#                 ) for _ in range(len(in_channels_list))]
#             )
#         else:
#             # 2mlp
#             self.scene_encoder = nn.Sequential(
#                 nn.Conv2d(scene_embedding_channels, out_channels, 1),
#                 nn.GroupNorm(32, out_channels),
#                 nn.ReLU(True),
#                 nn.Conv2d(out_channels, out_channels, 1),
#                 nn.GroupNorm(32, out_channels),
#                 nn.ReLU(True),
#             )
#             self.project = nn.Sequential(
#                 nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(True),
#                 nn.Dropout2d(p=0.1)
#             )
#
#         self.content_encoders = nn.ModuleList()
#         self.feature_reencoders = nn.ModuleList()
#         for c in in_channels_list:
#             self.content_encoders.append(
#                 nn.Sequential(
#                     nn.Conv2d(c, out_channels, 1),
#                     nn.BatchNorm2d(out_channels),
#                     nn.ReLU(True)
#                 )
#             )
#             self.feature_reencoders.append(
#                 nn.Sequential(
#                     nn.Conv2d(c, out_channels, 1),
#                     nn.BatchNorm2d(out_channels),
#                     nn.ReLU(True)
#                 )
#             )
#
#         self.normalizer = nn.Sigmoid()
#
#     def forward(self, scene_feature, features: list):
#         # [N, C, H, W]
#         content_feats = [c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)]
#         if self.scale_aware_proj:
#             scene_feats = [op(scene_feature) for op in self.scene_encoder]
#             relations = [self.normalizer((sf * cf).sum(dim=1, keepdim=True)) for sf, cf in
#                          zip(scene_feats, content_feats)]
#         else:
#             # [N, C, 1, 1]
#             scene_feat = self.scene_encoder(scene_feature)
#             relations = [self.normalizer((scene_feat * cf).sum(dim=1, keepdim=True)) for cf in content_feats]
#
#         p_feats = [op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)]
#
#         refined_feats = [torch.cat([r * p, o], dim=1) for r, p, o in zip(relations, p_feats, features)]
#
#         if self.scale_aware_proj:
#             ffeats = [op(x) for op, x in zip(self.project, refined_feats)]
#         else:
#             ffeats = [self.project(x) for x in refined_feats]
#
#         return ffeats

if __name__ == '__main__':
    img = torch.randn(2, 2048, 1, 1)
    feature_list = []
    for i in range(4):
        feature_list.append(torch.randn(2, 256 * 2 ** i, int(32 // (2 ** i)), int(32 // (2 ** i))))
    # model = FSRelation(2048, (256, 256, 256, 256), 256, scale_aware_proj=True)
    model = SceneRelation(2048, [256, 512, 1024, 2048],512)
    print(model(img, feature_list)[1].shape)