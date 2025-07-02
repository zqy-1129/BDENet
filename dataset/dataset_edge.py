"""
双时相多尺度变化检测数据集（带边缘）
"""
from PIL import Image
from torch.utils import data
from dataset.data_augmentation_edge import DataAugmentation

import os
import sys
import cv2 as cv
import numpy as np

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

"""
CD data set with pixel-level labels;
├─image_A
├─image_B
├─label
└─list
"""

IMG_A_FOLDER_NAME = 'A'
IMG_B_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
CD_FOLDER_NAME = 'CD_label'
EDGE_A_FOLDER_NAME = 'A_edge'
EDGE_B_FOLDER_NAME = 'B_edge'
EDGE_M_A_FOLDER_NAME = 'A_edge_pooling'
EDGE_M_B_FOLDER_NAME = 'B_edge_pooling'
IMG_A_LABEL_FOLDER_NAME = 'A_label'
IMG_B_LABEL_FOLDER_NAME = 'B_label'

IGNORE = 255

label_suffix = '.png'  # jpg for gan dataset, others : png


def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


def load_image_label_list_from_npy(npy_path, img_name_list):
    cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_img_A_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_A_FOLDER_NAME, img_name) + ".png"


def get_img_B_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_B_FOLDER_NAME, img_name) + ".png"


def get_edge_A_path(root_dir, img_name):
    return os.path.join(root_dir, EDGE_A_FOLDER_NAME, img_name) + ".png"


def get_edge_B_path(root_dir, img_name):
    return os.path.join(root_dir, EDGE_B_FOLDER_NAME, img_name) + ".png"


def get_edge_M_A_path(root_dir, img_name):
    return os.path.join(root_dir, EDGE_A_FOLDER_NAME, img_name) + ".png"


def get_edge_M_B_path(root_dir, img_name):
    return os.path.join(root_dir, EDGE_B_FOLDER_NAME, img_name) + ".png"


def get_A_label_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_A_LABEL_FOLDER_NAME, img_name) + ".png"


def get_B_label_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_B_LABEL_FOLDER_NAME, img_name) + ".png"


def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, CD_FOLDER_NAME, img_name) + '.png'


class ImageDataset(data.Dataset):
    def __init__(self, root_dir="D:\\study\\data\\podata", split='train', img_size=256, is_train=True, to_tensor=True,
                 muti_scale=False):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.muti_scale = muti_scale
        self.split = split  # train | train_aug | val
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split + '.txt')
        self.img_name_list_tmp = load_img_name_list(self.list_path)

        # 两种尺度名称列表，边界也可以直接用
        self.img_name_list_1 = []
        self.img_name_list_2 = []
        # 标签名称列表
        self.img_label_name_list = []

        for line in self.img_name_list_tmp:
            # 尺度一
            self.img_name_list_1.append(line + "_0")
            self.img_name_list_1.append(line + "_0" + "_aug_0")
            self.img_name_list_1.append(line + "_0" + "_aug_1")
            self.img_name_list_1.append(line + "_0" + "_aug_2")

            # 尺度二
            self.img_name_list_2.append(line + "_1")
            self.img_name_list_2.append(line + "_1" + "_aug_0")
            self.img_name_list_2.append(line + "_1" + "_aug_1")
            self.img_name_list_2.append(line + "_1" + "_aug_2")

            # 标签（尺度一）
            self.img_label_name_list.append(line)
            self.img_label_name_list.append(line + "_aug_0")
            self.img_label_name_list.append(line + "_aug_1")
            self.img_label_name_list.append(line + "_aug_2")

        self.size = len(self.img_name_list_1)
        self.to_tensor = to_tensor

        if is_train:
            self.augmentation = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
            )
        else:
            self.augmentation = DataAugmentation(
                img_size=self.img_size
            )

    def __getitem__(self, index):

        if self.muti_scale:
            name_1 = self.img_name_list_1[index]
            name_2 = self.img_name_list_2[index]

            A_path_1 = get_img_A_path(self.root_dir, self.img_name_list_1[index % self.size])
            B_path_1 = get_img_B_path(self.root_dir, self.img_name_list_1[index % self.size])
            A_path_2 = get_img_A_path(self.root_dir, self.img_name_list_2[index % self.size])
            B_path_2 = get_img_B_path(self.root_dir, self.img_name_list_2[index % self.size])
            edge_A_path = get_edge_A_path(self.root_dir, self.img_name_list_1[index % self.size])
            edge_B_path = get_edge_B_path(self.root_dir, self.img_name_list_1[index % self.size])
            edge_M_A_path = get_edge_M_A_path(self.root_dir, self.img_name_list_1[index % self.size])
            edge_M_B_path = get_edge_M_B_path(self.root_dir, self.img_name_list_1[index % self.size])

            img_A_1 = np.asarray(Image.open(A_path_1).convert('RGB'))
            img_B_1 = np.asarray(Image.open(B_path_1).convert('RGB'))
            img_A_2 = np.asarray(Image.open(A_path_2).convert('RGB'))
            img_B_2 = np.asarray(Image.open(B_path_2).convert('RGB'))
            edge_A = np.asarray(Image.open(edge_A_path).convert('RGB'))
            edge_B = np.asarray(Image.open(edge_B_path).convert('RGB'))
            edge_M_A = np.asarray(Image.open(edge_M_A_path).convert('RGB'))
            edge_M_B = np.asarray(Image.open(edge_M_B_path).convert('RGB'))

            [img_A_1, img_B_1], _ = self.augmentation.transform([img_A_1, img_B_1], [], to_tensor=self.to_tensor)
            [img_A_2, img_B_2], _ = self.augmentation.transform([img_A_2, img_B_2], [], to_tensor=self.to_tensor)
            [edge_A, edge_B], _ = self.augmentation.transform([edge_A, edge_B], [], to_tensor=self.to_tensor)
            [edge_M_A, edge_M_B], _ = self.augmentation.transform([edge_M_A, edge_M_B], [], to_tensor=self.to_tensor)

            return {'A_1': img_A_1, 'A_2': img_A_2, 'edge_A': edge_A, 'edge_M_A': edge_M_A,
                    'B_1': img_B_1, 'B_2': img_B_2, 'edge_B': edge_B, 'edge_M_B': edge_M_B,
                    'name': name_1, 'name_2': name_2}

        else:
            name = self.img_name_list_1[index]

            A_path = get_img_A_path(self.root_dir, self.img_name_list_1[index % self.size])
            B_path = get_img_B_path(self.root_dir, self.img_name_list_1[index % self.size])
            edge_A_path = get_edge_A_path(self.root_dir, self.img_name_list_1[index % self.size])
            edge_B_path = get_edge_B_path(self.root_dir, self.img_name_list_1[index % self.size])
            edge_M_A_path = get_edge_M_A_path(self.root_dir, self.img_name_list_1[index % self.size])
            edge_M_B_path = get_edge_M_B_path(self.root_dir, self.img_name_list_1[index % self.size])

            img_A = np.asarray(Image.open(A_path).convert('RGB'))
            img_B = np.asarray(Image.open(B_path).convert('RGB'))
            edge_A = np.asarray(Image.open(edge_A_path).convert('RGB'))
            edge_B = np.asarray(Image.open(edge_B_path).convert('RGB'))
            edge_M_A = np.asarray(Image.open(edge_M_A_path).convert('RGB'))
            edge_M_B = np.asarray(Image.open(edge_M_B_path).convert('RGB'))

            [img_A, img_B], _ = self.augmentation.transform([img_A, img_B], [], to_tensor=self.to_tensor)
            [edge_A, edge_B], _ = self.augmentation.transform([edge_A, edge_B], [], to_tensor=self.to_tensor)
            [edge_M_A, edge_M_B], _ = self.augmentation.transform([edge_M_A, edge_M_B], [], to_tensor=self.to_tensor)

            return {'A': img_A, 'B': img_B, 'edge_A': edge_A, 'edge_M_A': edge_M_A,
                    'edge_B': edge_B, 'edge_M_B': edge_M_B, 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.size


class Dataset(ImageDataset):
    def __init__(self, root_dir="D:\\study\\data\\podata", img_size=256, split='train', is_train=True,
                 label_transform=None, to_tensor=True, muti_scale=False):
        super(Dataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train, to_tensor=to_tensor)
        self.label_transform = label_transform
        self.muti_scale = muti_scale

    def __getitem__(self, index):
        if self.muti_scale:
            name_1 = self.img_name_list_1[index]
            name_2 = self.img_name_list_2[index]

            A_path_1 = get_img_A_path(self.root_dir, self.img_name_list_1[index % self.size])
            B_path_1 = get_img_B_path(self.root_dir, self.img_name_list_1[index % self.size])
            A_path_2 = get_img_A_path(self.root_dir, self.img_name_list_2[index % self.size])
            B_path_2 = get_img_B_path(self.root_dir, self.img_name_list_2[index % self.size])
            edge_A_path = get_edge_A_path(self.root_dir, self.img_name_list_1[index % self.size])
            edge_B_path = get_edge_B_path(self.root_dir, self.img_name_list_1[index % self.size])
            edge_M_A_path = get_edge_M_A_path(self.root_dir, self.img_name_list_1[index % self.size])
            edge_M_B_path = get_edge_M_B_path(self.root_dir, self.img_name_list_1[index % self.size])

            img_A_1 = np.asarray(Image.open(A_path_1).convert('RGB'))
            img_B_1 = np.asarray(Image.open(B_path_1).convert('RGB'))
            img_A_2 = np.asarray(Image.open(A_path_2).convert('RGB'))
            img_B_2 = np.asarray(Image.open(B_path_2).convert('RGB'))
            edge_A = np.asarray(Image.open(edge_A_path).convert('RGB'))
            edge_B = np.asarray(Image.open(edge_B_path).convert('RGB'))
            edge_M_A = np.asarray(Image.open(edge_M_A_path).convert('RGB'))
            edge_M_B = np.asarray(Image.open(edge_M_B_path).convert('RGB'))

            L_path = get_label_path(self.root_dir, self.img_label_name_list[index % self.size])
            img_L = cv.imread(L_path)
            imgGray = cv.cvtColor(img_L, cv.COLOR_BGR2GRAY)
            label = np.array(imgGray, dtype=np.uint8)

            L_A_path = get_A_label_path(self.root_dir, self.img_label_name_list[index % self.size])
            img_A_L = cv.imread(L_A_path)
            imgGrayA = cv.cvtColor(img_A_L, cv.COLOR_BGR2GRAY)
            label_A = np.array(imgGrayA, dtype=np.uint8)

            L_B_path = get_B_label_path(self.root_dir, self.img_label_name_list[index % self.size])
            img_B_L = cv.imread(L_B_path)
            imgGrayB = cv.cvtColor(img_B_L, cv.COLOR_BGR2GRAY)
            label_B = np.array(imgGrayB, dtype=np.uint8)

            #  二分类中，前景标注为255
            if self.label_transform == 'norm':
                label = label // 255

            [img_A_1, img_B_1], [img_A_2, img_B_2], [edge_A, edge_B], [edge_M_A, edge_M_B], [label], [label_A], \
                [label_B] = self.augmentation.transform_muti_scale(
                [img_A_1, img_B_1], [img_A_2, img_B_2], [edge_A, edge_B], [edge_M_A, edge_M_B], [label], [label_A],
                [label_B], to_tensor=self.to_tensor)
            return img_A_1, img_A_2, edge_A, edge_M_A, label_A, img_B_1, img_B_2, edge_B, edge_M_B, label_B, label

        else:
            name = self.img_name_list_1[index]

            A_path = get_img_A_path(self.root_dir, self.img_name_list_1[index % self.size])
            B_path = get_img_B_path(self.root_dir, self.img_name_list_1[index % self.size])
            edge_A_path = get_edge_A_path(self.root_dir, self.img_name_list_1[index % self.size])
            edge_B_path = get_edge_B_path(self.root_dir, self.img_name_list_1[index % self.size])
            edge_M_A_path = get_edge_M_A_path(self.root_dir, self.img_name_list_1[index % self.size])
            edge_M_B_path = get_edge_M_B_path(self.root_dir, self.img_name_list_1[index % self.size])

            img_A = np.asarray(Image.open(A_path).convert('RGB'))
            img_B = np.asarray(Image.open(B_path).convert('RGB'))
            edge_A = np.asarray(Image.open(edge_A_path).convert('RGB'))
            edge_B = np.asarray(Image.open(edge_B_path).convert('RGB'))
            edge_M_A = np.asarray(Image.open(edge_M_A_path).convert('RGB'))
            edge_M_B = np.asarray(Image.open(edge_M_B_path).convert('RGB'))

            L_path = get_label_path(self.root_dir, self.img_label_name_list[index % self.size])
            img_L = cv.imread(L_path)
            imgGray = cv.cvtColor(img_L, cv.COLOR_BGR2GRAY)
            label = np.array(imgGray, dtype=np.uint8)

            L_A_path = get_A_label_path(self.root_dir, self.img_label_name_list[index % self.size])
            img_A_L = cv.imread(L_A_path)
            imgGrayA = cv.cvtColor(img_A_L, cv.COLOR_BGR2GRAY)
            label_A = np.array(imgGrayA, dtype=np.uint8)

            L_B_path = get_B_label_path(self.root_dir, self.img_label_name_list[index % self.size])
            img_B_L = cv.imread(L_B_path)
            imgGrayB = cv.cvtColor(img_B_L, cv.COLOR_BGR2GRAY)
            label_B = np.array(imgGrayB, dtype=np.uint8)

            if self.label_transform == 'norm':
                label = label // 255

            [img_A, img_B], [edge_A, edge_B], [edge_M_A, edge_M_B], [label], [label_A], [label_B] = \
                self.augmentation.transform(
                    [img_A, img_B], [edge_A, edge_B],
                    [edge_M_A, edge_M_B], [label],
                    [label_A], [label_B], to_tensor=self.to_tensor)
            return img_A, edge_A, edge_M_A, label_A, img_B, edge_B, edge_M_B, label_B, label