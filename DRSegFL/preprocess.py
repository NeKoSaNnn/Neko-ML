#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import os.path as osp
import sys

import numpy as np
import torch
from PIL import Image

root_dir_name = osp.dirname(sys.path[0])  # ...Neko-ML/
now_dir_name = sys.path[0]  # ...DRSegFL/
sys.path.append(root_dir_name)

from DRSegFL import utils
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def ISIC_preprocess(img_path, target_path, img_size):
    """
    :param img_path:
    :param target_path:
    :param img_size:
    :return: tensor_img [Channel,H,W] ; tensor_target [1,H,W]:values in [0,1] ; pil_img ; pil_target
    """
    img = Image.open(img_path)
    img_trans = transforms.Compose([
        transforms.CenterCrop(max(img.size)),
        transforms.Resize((img_size, img_size))
    ])
    pil_img = img_trans(img)
    tensor_img = transforms.ToTensor()(pil_img)

    if utils.is_img(target_path):
        target = Image.open(target_path).convert("L")
        target_trans = transforms.Compose([
            transforms.CenterCrop(max(img.size)),
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST)
        ])
        pil_target = target_trans(target)
        tensor_target = torch.from_numpy(np.asarray(pil_target, dtype=np.long))
        tensor_target[tensor_target > 0] = 1
        tensor_target = tensor_target.unsqueeze(0)
    else:
        raise InterruptedError("标签数据非图片数据，需要额外处理")
    return tensor_img, tensor_target, pil_img, pil_target


def DDR_preprocess(img_path, target_path, img_size):
    """
    :param img_path:
    :param target_path:
    :param img_size:
    :return: tensor_img [Channel,H,W] ; tensor_target [H,W]:values in [0,num_classes); pil_img ; pil_target
    """

    # foreground crop
    # fore_h, fore_w = utils.get_foreground_hw(img_path)
    # pad_h, pad_w = max(0, (fore_w - fore_h) // 2), max(0, (fore_h - fore_w) // 2)
    # trans = transforms.Compose([
    #     transforms.CenterCrop((fore_h, fore_w)),
    #     transforms.Pad((pad_w, pad_h)),
    #     transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC)
    # ])
    img = Image.open(img_path)
    img_trans = transforms.Compose([
        transforms.CenterCrop(min(img.size)),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC)
    ])
    pil_img = img_trans(img)
    tensor_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4211, 0.2640, 0.1104], std=[0.3133, 0.2094, 0.1256])
    ])(pil_img)

    if utils.is_img(target_path):
        target = Image.open(target_path)
        target_trans = transforms.Compose([
            transforms.CenterCrop(min(img.size)),
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST)
        ])
        pil_target = target_trans(target)
        tensor_target = torch.from_numpy(np.asarray(pil_target, dtype=np.long))
    else:
        raise InterruptedError("标签数据非图片数据，需要额外处理")
    return tensor_img, tensor_target, pil_img, pil_target
