#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import random
from os import path as osp

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from utils.utils import utils

utils = utils(log_path="./log")


class SplitDataSet(Dataset):
    def __init__(self, dataset, idxs):
        assert isinstance(idxs, list)
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        return self.dataset[self.idxs[index]]


class ISICDataSet(Dataset):
    def __init__(self, args, root, dataset_type, dataset_list_txt):
        assert dataset_type == "train" or dataset_type == "val" or dataset_type == "test"
        self.args = args
        self.dataset_type = dataset_type
        self.dataset = []
        path = osp.join(root, dataset_type)

        self.trans = transforms.ToTensor()
        self.transTarget = transforms.ToTensor()

        data_path = osp.join(path, "image")
        target_path = osp.join(path, "mask")

        with open(osp.join(root, dataset_list_txt)) as f:
            for name in f.readlines():
                self.dataset.append([osp.join(data_path, name.strip() + ".jpg"),
                                     osp.join(target_path, name.strip() + "_segmentation.png")])
        random.shuffle(self.dataset)

    @classmethod
    def preprocess(cls, data_path: str, to_gray=False):
        ndimg = cv.imread(data_path)
        if to_gray:
            ndimg = cv.cvtColor(ndimg, cv.COLOR_BGR2GRAY)
        newW, newH = 256, 256
        ndimg = cv.resize(ndimg, (newW, newH), interpolation=cv.INTER_LINEAR)

        if len(ndimg.shape) == 2:
            ndimg = np.expand_dims(ndimg, axis=2)

        # HWC to CHW
        transimg = ndimg.transpose((2, 0, 1))
        transimg = transimg / 255 if np.max(transimg) > 1 else transimg
        return transimg

    @classmethod
    def data_preprocess(cls, data_path: str, tran=None):
        data = cv.imread(data_path, flags=cv.IMREAD_COLOR)
        data = cv.resize(data, [256, 256], interpolation=cv.INTER_LINEAR)
        data = data / 255 if np.max(data) > 1 else data
        # data = trans(data) if trans else data
        # (H,W,C) to (C,H,W)
        data = data.transpose((2, 0, 1))
        return data

    @classmethod
    def target_preprocess(cls, target_path: str, transTarget=None, num_classes=1):
        target = cv.imread(target_path, flags=cv.IMREAD_GRAYSCALE)
        target = cv.resize(target, [256, 256], interpolation=cv.INTER_LINEAR)
        # target = transTarget(target) if transTarget else target
        target = target / 255 if np.max(target) > 1 else target
        # (H,W,C) to (C,H,W)
        target = target.transpose((2, 0, 1))
        return target
        # # isic数据集若两分类，以0.5为界分类
        # if num_classes == 1:
        #     # 分类后target为3维tensor
        #     # 不需要one_hot
        #     # return (target > 0.5).float()
        #     return target
        # elif num_classes == 2:
        #     # 分类后target为2维tensor
        #     # 方便one_hot计算
        #     return (target.squeeze() > 0.5).long()
        # return target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_path, target_path = self.dataset[index]
        data, target = ISICDataSet.preprocess(data_path), ISICDataSet.preprocess(target_path, to_gray=True)
        # data, target = ISICDataSet.data_preprocess(data_path, self.trans), \
        #                ISICDataSet.target_preprocess(target_path, self.transTarget, num_classes=self.args.num_classes)
        data, target = torch.from_numpy(data), torch.from_numpy(target)
        return data, target


class InitDataSet(object):
    def __init__(self, args, trans=None, transTarget=None, dataset_path="/data"):
        self.args = args
        self.dataset_path = dataset_path
        self.trans = trans
        self.transTarget = transTarget
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        # np.random.seed(self.args.seed)

    def get(self):
        trans = self.trans
        transTarget = self.transTarget

        if self.args.dataset == "mnist":
            self.train_dataset = datasets.MNIST(root=self.dataset_path, train=True, transform=trans, download=True)
            self.test_dataset = datasets.MNIST(root=self.dataset_path, train=False, transform=trans, download=True)
        elif self.args.dataset == "cifar10":
            self.train_dataset = datasets.CIFAR10(root=self.dataset_path, train=True, transform=trans, download=True)
            self.test_dataset = datasets.CIFAR10(root=self.dataset_path, train=False, transform=trans, download=True)
        elif self.args.dataset == "isic":
            self.train_dataset = ISICDataSet(args=self.args, root=osp.join(self.dataset_path, "ISIC"),
                                             dataset_type="train", dataset_list_txt="train.txt")
            self.val_dataset = ISICDataSet(args=self.args, root=osp.join(self.dataset_path, "ISIC"), dataset_type="val",
                                           dataset_list_txt="val.txt")
            self.test_dataset = ISICDataSet(args=self.args, root=osp.join(self.dataset_path, "ISIC"),
                                            dataset_type="test", dataset_list_txt="test.txt")
        else:
            exit("Unable to identify the dataset")
        utils.log("train dataset", {"size": None if self.train_dataset is None else len(self.train_dataset)})
        utils.log("val dataset", {" size": None if self.val_dataset is None else len(self.val_dataset)})
        utils.log("test dataset", {" size": None if self.test_dataset is None else len(self.test_dataset)})
        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_iid_user_dataidx(self, dataset):
        # 按 num_users 对dataset进行分割，模拟独立同分布训练数据
        assert self.args.iid and self.args.num_users >= 1
        per_user_dataset_nums = len(dataset) // self.args.num_users
        user_dataidxs, all_dataidxs = dict(), list(range(len(dataset)))
        for u in range(self.args.num_users):
            # replace=False 不取相同值
            user_dataidxs[u] = list(np.random.choice(all_dataidxs, per_user_dataset_nums, replace=False))
            all_dataidxs = list(set(all_dataidxs) - set(user_dataidxs[u]))
        return user_dataidxs

    def get_dataloader(self):
        return DataLoader(self.train_dataset, self.args.train_bs, shuffle=True), \
               DataLoader(self.val_dataset, self.args.test_bs, shuffle=False), \
               DataLoader(self.test_dataset, self.args.test_bs, shuffle=False)

    def get_iid_dataloader(self, dataset, idx):
        return DataLoader(SplitDataSet(dataset, idx), batch_size=self.args.local_bs, shuffle=True)

    def get_non_iid_user_dataidx(self, dataset):
        return None


if __name__ == "__main__":
    img = Image.open("/home/maojingxin/workspace/Neko-ML/FL/data/ISIC/test/mask/ISIC_0000047_segmentation.png")
    print(img.mode)
    print(img.size)
    print(img.getpixel((0, 0)))

    tensor_img = transforms.ToTensor()(img)
    print(tensor_img.shape)
    print(tensor_img[0][0])

    P_img = img.convert("P")
    print(P_img.mode)
    print(P_img.size)
    print(P_img.getpixel((0, 0)))

    tensor_P_img = transforms.ToTensor()(P_img)
    print(tensor_P_img.shape)
    print(tensor_P_img[0][0])

    P_array = np.asarray(P_img)
    print(P_array.shape)
    print(P_array[0][0])
