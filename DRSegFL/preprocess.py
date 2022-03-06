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


def ISIC_preprocess(img_path, target_path, img_size):
    """
    :param img_path:
    :param target_path:
    :param img_size:
    :return: tensor_img [Channel,H,W] ; tensor_target [1,H,W]:values in [0,1]
    """
    _, tensor_img = utils.to_tensor_use_pil(img_path, img_size)

    if utils.is_img(target_path):
        target = Image.open(target_path).convert("L").resize((img_size, img_size))
        np_target = np.asarray(target, dtype=np.float)
        np_target[np_target > 0] = 1
        tensor_target = torch.from_numpy(np_target).unsqueeze(0)
        # tensor_target = utils.ignore_background(tensor_target, self.num_classes, 0)
        # _, tensor_target = utils.to_tensor_use_pil(target_path, self.img_size, to_gray=True)
    else:
        raise InterruptedError("标签数据非图片数据，需要额外处理")
    return tensor_img, tensor_target


def DDR_preprocess(img_path, target_path, img_size, num_classes):
    """
    :param img_path:
    :param target_path:
    :param img_size:
    :param num_classes:
    :return: tensor_img [Channel,H,W], tensor_target [H,W]:values in [0,num_classes],ignore_index=num_classes
    """
    _, tensor_img = utils.to_tensor_use_pil(img_path, img_size)

    if utils.is_img(target_path):
        target = Image.open(target_path).resize((img_size, img_size))
        tensor_target = torch.from_numpy(np.asarray(target, dtype=np.long))
        tensor_target = utils.ignore_background(tensor_target, num_classes, 0)
        # tensor_target = utils.make_one_hot(tensor_target, self.num_classes)
    else:
        raise InterruptedError("标签数据非图片数据，需要额外处理")
    return tensor_img, tensor_target
