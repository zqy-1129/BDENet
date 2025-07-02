import random
import os
import sys
import time
import argparse as ag

from functools import reduce
from models.BiSRNet import *
from models.ClearSCD import *
from models.MySCD import *
from models.MySCD_V1 import *
from models.RegularLandSCD import ISwinUperNetV5
from models.RegularLandSCDV2 import *
from models.SwinUNet import *
from models.swin_transformer import *


sys.path.append("..")  # 将上级目录添加到模块搜索路径中，允许从models目录导入模型。


def seed_torch(seed=6):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def count_model_parameters(module, _default_logger=None):
    cnt = 0
    for p in module.parameters():
        cnt += reduce(lambda x, y: x * y, list(p.shape))
    print('#params: {}, {} M'.format(cnt, round(cnt / float(1e6), 3)))
    return cnt


def measure_inference_speed(model, data, max_iter=200, log_interval=50):
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i in range(max_iter):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(*data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    return fps


def get_model(args):
    if args.model == 'BiSRNet':
        return BiSRNet(num_classes=6)
    elif args.model == 'ClearSCD':
        return ClearSCD(args)
    elif args.model == 'RegularLandSCD':
        return ISwinUperNetV5(pretrain_img_size=args.size, num_classes=args.num_segclass, in_chans=args.num_channel,
                              use_attens=1)
    elif args.model == "BDSANet":
        return BDSANet(num_class=args.num_channel)
    elif args.model == "BDSANet_V1":
        return BDSANet_V1(num_class=args.num_channel)
    elif args.model == "only_swin":
        return OnlySwinTransformer(num_classes=args.num_channel)
    elif args.model == "SwinUNet":
        return SwinUNet(num_classes=args.num_channel)


if __name__ == '__main__':
    parser = ag.ArgumentParser(description='Training change detection network')
    parser.add_argument("--model", type=str, default="BiSRNet")
    parser.add_argument("--backbone", type=str, default="efficientnet-b0")
    parser.add_argument("--num_segclass", type=int, default=82)
    parser.add_argument("--num_channel", type=int, default=3)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--queue_len", type=int, default=512)
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--fusion", type=str, default="diff")
    parser.add_argument("--downsample_seg", type=bool, default=False)
    parser.add_argument("--bcd_convs_num", type=int, default=3)
    parser.add_argument("--temperature", type=int, default=9)
    parser.add_argument("--exchange_features", type=bool, default=False)
    parser.add_argument("--only_bcd", type=bool, default=False)
    parser.add_argument("--only_seg", type=bool, default=False)
    parser.add_argument("--with_bcl", type=bool, default=False)
    parser.add_argument("--with_cl", type=bool, default=False)
    parser.add_argument("--seg_pretrain", type=bool, default=False)
    parser.add_argument("--mode", type=str, default='from_to')
    parser.add_argument("--inner_channels", type=int, default=128)
    parser.add_argument("--num_convs", type=int, default=4)
    args = parser.parse_args()

    from thop import profile
    from thop import clever_format

    model = get_model(args).cuda()
    # print(model.decoder.merge)
    input = torch.randn(1, 6, 256, 256).cuda()

    macs, params = profile(model, inputs=(input,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)

    count_params = count_model_parameters(model)
    print(count_params)

    input = torch.randn(1, 6, 256, 256).cuda()
    fps = measure_inference_speed(model, (input,))
    print(fps)
