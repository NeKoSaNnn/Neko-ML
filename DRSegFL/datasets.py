#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""

import glob
import os.path as osp

import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from DRSegFL import utils


# datas in datas_dir and targets in targets_dir
def dataset_txt_generate(datas_dir: str, data_suffix: str, targets_dir: str, target_suffix: str, txt_file_path: str):
    datas = sorted(glob.glob(osp.join(datas_dir, "*.{}".format(data_suffix))))
    targets = sorted(glob.glob(osp.join(targets_dir, "*.{}".format(target_suffix))))
    assert len(datas) == len(targets), "len(datas):{} != len(targets):{}".format(len(datas), len(targets))
    lines = [datas[i] + " " + targets[i] + "\n" for i in range(len(datas))]

    with open(txt_file_path, "w+") as f:
        f.writelines(lines)


# datas in datas_dir and targets in targets_dir
def iid_dataset_txt_generate(datas_dir: str, data_suffix: str, targets_dir: str, target_suffix: str,
                             txt_file_paths: list):
    datas = sorted(glob.glob(osp.join(datas_dir, "*.{}".format(data_suffix))))
    targets = sorted(glob.glob(osp.join(targets_dir, "*.{}".format(target_suffix))))
    assert len(datas) == len(targets), "len(datas):{} != len(targets):{}".format(len(datas), len(targets))
    lines = [datas[i] + " " + targets[i] + "\n" for i in range(len(datas))]

    split_num = len(txt_file_paths)
    per_split_data_size = len(lines) // split_num
    for i in range(split_num - 1):
        now_split_lines = np.random.choice(lines, per_split_data_size, replace=False)
        lines = list(set(lines) - set(now_split_lines))
        with open(txt_file_paths[i], "w+") as f:
            f.writelines(now_split_lines)
    with open(txt_file_paths[-1], "w+") as f:
        f.writelines(lines)


class ListDataset(Dataset):
    def __init__(self, txt_path: str, img_size=256, is_augment=False):
        with open(txt_path, "r") as f:
            self.datas_and_targets_path = f.readlines()
        self.datas_and_targets_path = [data_and_target.strip().split(" ", 2) for data_and_target in
                                       self.datas_and_targets_path]

        self.img_size = img_size
        self.is_augment = is_augment

    def __getitem__(self, index):
        img_path = self.datas_and_targets_path[index][0].strip()
        target_path = self.datas_and_targets_path[index][1].strip()

        img = Image.open(img_path).convert("RGB")
        tensor_img = transforms.ToTensor()(img)
        tensor_img = F.interpolate(tensor_img.unsqueeze(0), self.img_size, mode="nearest").squeeze(0)

        if utils.is_img(target_path):
            target = Image.open(target_path)
            tensor_target = transforms.ToTensor()(target)
            tensor_target = F.interpolate(tensor_target.unsqueeze(0), self.img_size, mode="nearest").squeeze(0)
        else:
            raise InterruptedError("标签数据非图片数据，需要额外处理")

        return tensor_img, tensor_target, img_path

    def __len__(self):
        return len(self.datas_and_targets_path)
