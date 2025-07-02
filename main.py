from argparse import ArgumentParser
from dataset.dataset import Dataset
from dataset.dataset_edge import Dataset as Dataset_edge

import torch
from torch.utils.data import DataLoader
import os
from trainer import CDTrainer
from evaluatorSCD import CDEvaluator


def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])


def train(args):
    torch.cuda.empty_cache()
    if args.edge:
        train_sets = Dataset_edge(split='train', muti_scale=args.muti_scale)
        val_sets = Dataset_edge(split='val', muti_scale=args.muti_scale)
    else:
        train_sets = Dataset(split='train', muti_scale=args.muti_scale)
        val_sets = Dataset(split='val', muti_scale=args.muti_scale)
    datasets = {'train': train_sets, 'val': val_sets}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, drop_last=True, shuffle=True,
                                 num_workers=args.num_workers) for x in ['train', 'val']}
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()


def test(args):
    if args.edge:
        test_sets = Dataset_edge(split='train', is_train=False, muti_scale=args.muti_scale)
    else:
        test_sets = Dataset(split='train', is_train=False, muti_scale=args.muti_scale)
    file_list = test_sets.img_name_list_1
    file_label_list = test_sets.img_label_name_list
    dataloader = DataLoader(test_sets, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = CDEvaluator(args=args, dataloader=dataloader, file_list=file_list, file_label_list=file_label_list)

    model.eval_models()


if __name__ == "__main__":
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--train_model', default='SwinV2_AllModules_V1', type=str)
    parser.add_argument('--project_name', default='BDENet', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints/', type=str)
    parser.add_argument("--road_path", default="D:\\study\\data\\podata", type=str)

    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='Dataset', type=str)
    parser.add_argument('--data_name', default='GEP', type=str)
    parser.add_argument('--num_channel', default=3, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--edge', default=True, type=bool)
    parser.add_argument('--muti_scale', default=False, type=bool)

    # model
    parser.add_argument('--n_class', default=6, type=int)
    parser.add_argument('--loss', default='ce', type=str)

    # optimizer
    parser.add_argument('--doptimizer', default='adam', type=str)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_epochs', default=300, type=int)
    parser.add_argument('--lr_policy', default='poly', type=str, help='linear | step')
    parser.add_argument('--lr_decay_iters', default=30, type=int)

    args = parser.parse_args()

    get_device(args)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # train(args)

    test(args)

