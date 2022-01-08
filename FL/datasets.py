#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class SplitDataSet(Dataset):
    def __init__(self, dataset, idxs):
        assert isinstance(idxs, list)
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


class InitDataSet(object):
    def __init__(self, args, dataset_path="/data", trans=transforms.ToTensor()):
        self.args = args
        self.dataset_path = dataset_path
        self.trans = [trans]
        self.train_dataset = None
        self.test_dataset = None
        # np.random.seed(self.args.seed)

    def addTrans(self, transform):
        self.trans.append(transform)

    def get(self):
        trans = transforms.Compose(self.trans)

        if self.args.dataset == "mnist":
            self.train_dataset = datasets.MNIST(root=self.dataset_path, train=True, transform=trans, download=True)
            self.test_dataset = datasets.MNIST(root=self.dataset_path, train=False, transform=trans, download=True)
        elif self.args.dataset == "cifar10":
            self.train_dataset = datasets.CIFAR10(root=self.dataset_path, train=True, transform=trans, download=True)
            self.test_dataset = datasets.CIFAR10(root=self.dataset_path, train=False, transform=trans, download=True)
        else:
            exit("Unable to identify the dataset")
        return self.train_dataset, self.test_dataset

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
               DataLoader(self.test_dataset, self.args.test_bs, shuffle=False)

    def get_iid_dataloader(self, dataset, idx):
        return DataLoader(SplitDataSet(dataset, idx), batch_size=self.args.local_bs, shuffle=True)

    def get_non_iid_user_dataidx(self, dataset):
        return None
