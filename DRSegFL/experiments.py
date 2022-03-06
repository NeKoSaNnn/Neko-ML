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

# path = "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/train/annotation/007-6361-400.png"
# img = Image.open(path)
#
# color_map = imgviz.label_colormap()
# nd_img = np.asarray(img)
# print(nd_img.shape)
# torch_img = torch.from_numpy(nd_img.astype(np.int64))
# print(torch_img.shape)
# torch_img = torch_img.unsqueeze(0)
# print(torch_img.shape)
# print(color_map)
#
# print(img.size)
# print(img.getpixel((1000, 1500)))
# print(img.getpixel((1000, 1501)))
# print(img.getpixel((1000, 1502)))
# print(img.getpixel((1000, 1503)))
#
# a = np.asarray(img)
# print(a.shape)
# print(a[1500][1000])
# print(a[1501][1000])
# print(a[1502][1000])
# print(a[1503][1000])

# onehot_img = utils.make_one_hot(torch_img, 4)
# print(onehot_img.shape)
# onehot_img = onehot_img.squeeze(0)
# print(torch.max(onehot_img[0]))
# print(torch.max(onehot_img[1]))
# print(torch.max(onehot_img[2]))
# print(torch.max(onehot_img[3]))

a = np.array([[1, 2, 1],
              [0, 1, 1],
              [0, 0, 2]])
torch_a = torch.from_numpy(a)
print(a.shape)
one_hot_a = utils.make_one_hot(torch_a, 2)
print(one_hot_a)
print(one_hot_a.shape)

b = np.array([[[1, 2, 1],
               [0, 1, 1],
               [0, 0, 2]]])
print(b.shape)
print(b.dtype)
torch_b = torch.from_numpy(b)
print(torch_b.shape)
print(torch_b.dtype)
one_hot_b = utils.batch_make_one_hot(torch_b, 2)
print(one_hot_b)
print(one_hot_b.shape)
print(one_hot_b.dtype)

c = np.array([[[1, 2, 1],
              [0, 1, 1],
              [0, 0, 2]]])
c = utils.ignore_background(c, 2, 1)
mask = (c != 0)
C = c[mask]
print(mask)
print(c)
print(C)

d = np.array([[[1, 2, 1],
               [0, 1, 1],
               [0, 0, 2]],
              [[0, 0, 1],
               [0, 1, 1],
               [0, 0, 0]]
              ])
D = []
D.extend(d[:, ])
print(D)
