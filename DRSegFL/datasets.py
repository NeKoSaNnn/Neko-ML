#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""

import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import utils


class ListDataset(Dataset):
    def __init__(self, txt_path: str, img_type: str, target_type: str, img_size=256, is_augment=False):
        with open(txt_path, "r") as f:
            self.img_files_path = f.readlines()
        self.target_files_path = [path.replace("imgs", "targets").replace(img_type, target_type) for path in
                                  self.img_files_path]
        self.img_type = img_type
        self.target_type = target_type
        self.img_size = img_size
        self.is_augment = is_augment

    def __getitem__(self, index):
        img_path = self.img_files_path[index].strip()
        target_path = self.target_files_path[index].strip()

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
        return len(self.img_files_path)
