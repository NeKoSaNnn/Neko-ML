#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""

import cv2 as cv
import numpy as np
import imgviz
import torch
from PIL import Image
import utils
from torch.nn import functional as F


def _make_one_hot(label, num_classes):
    """
    :param label: [N, *], values in [0,num_classes)
    :return: [N, C, *]
    """
    label = label.unsqueeze(1)
    shape = list(label.shape)
    shape[1] = num_classes

    result = torch.zeros(shape, device=label.device)
    result.scatter_(1, label, 1)

    return result


path = "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/train/annotation/007-6361-400.png"
img = Image.open(path)

color_map = imgviz.label_colormap()
nd_img = np.asarray(img)
print(nd_img.shape)
torch_img = torch.from_numpy(nd_img.astype(np.int64))
print(torch_img.shape)
torch_img = torch_img.unsqueeze(0)
print(torch_img.shape)
onehot_img = _make_one_hot(torch_img, 4)
print(color_map)

print(img.size)
print(img.getpixel((1000, 1500)))
print(img.getpixel((1000, 1501)))
print(img.getpixel((1000, 1502)))
print(img.getpixel((1000, 1503)))

a = np.asarray(img)
print(a.shape)
print(a[1500][1000])
print(a[1501][1000])
print(a[1502][1000])
print(a[1503][1000])

print(onehot_img.shape)
onehot_img = onehot_img.squeeze(0)
print(torch.max(onehot_img[0]))
print(torch.max(onehot_img[1]))
print(torch.max(onehot_img[2]))
print(torch.max(onehot_img[3]))
