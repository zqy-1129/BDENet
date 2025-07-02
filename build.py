from models.BiSRNet import BiSRNet
from models.HGINet import HGINet
from models.HRSCD4 import HRSCD4
from models.MTSCD_Net import MTSCDNet
from models.SCanNet import SCanNet

from models.FC_EF import FC_EF
from models.FC_Siam_Conv import FC_Siam_Conv
from models.FC_Siam_Diff import FC_Siam_Diff
from models.UNetPlusPlus import UNetPlusPlus

from models.SwinV2_AllModules_V1 import SwinV2_UpperHead_AllModules_V1


class Builder(object):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.models = {
            "BiSRNet": BiSRNet,
            "HGINet": HGINet,
            "HRSCD4": HRSCD4,
            "MTSCDNet": MTSCDNet,
            "SCanNet": SCanNet,

            "FC_EF": FC_EF,
            "FC_Siam_Conv": FC_Siam_Conv,
            "FC_Siam_Diff": FC_Siam_Diff,
            "UNetPlusPlus": UNetPlusPlus,

            "SwinV2_AllModules_V1": SwinV2_UpperHead_AllModules_V1
        }

    def build_model(self):
        if self.args.train_model is None or self.args.train_model not in self.models:
            raise NotImplementedError
        model = self.models[self.args.train_model]
        if model in (BiSRNet,):
            return model(num_classes=self.args.n_class)
        elif model in (HGINet,):
            return model(num_classes=self.args.n_class)
        elif model in (HRSCD4,):
            return model(input_nbr=3, label_nbr=self.args.n_class, wsl=True)
        elif model in (MTSCDNet,):
            return model('swin_small', False, 6, False)
        elif model in (SCanNet,):
            return model(in_channels=self.args.num_channel, num_classes=self.args.n_class,
                         input_size=self.args.img_size)

        elif model in (FC_EF,):
            return model(num_classes=self.args.n_class)
        elif model in (FC_Siam_Conv,):
            return model(num_classes=self.args.n_class)
        elif model in (FC_Siam_Diff,):
            return model(num_classes=self.args.n_class)
        elif model in (UNetPlusPlus,):
            return model(num_classes=self.args.n_class)

        elif model in (SwinV2_UpperHead_AllModules_V1,):
            return model(num_classes=self.args.n_class)

        else:
            return model(num_classes=self.args.n_class,
                         backbone=self.args.encoder,
                         output_stride=self.args.out_stride,
                         sync_bn=self.args.sync_bn,
                         freeze_bn=self.args.freeze_bn)
