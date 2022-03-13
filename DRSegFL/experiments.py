#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import logging

import cv2 as cv
import numpy as np
import imgviz
import torch
from PIL import Image
from torchvision import transforms

import utils, preprocess, datasets
from logger import Logger
from loss import *
from torch.nn import functional as F
import segmentation_models_pytorch as smp
import torch.nn as nn

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

# a = np.array([[1, 2, 1],
#               [0, 1, 1],
#               [0, 0, 2]])
# torch_a = torch.from_numpy(a)
# print(a.shape)
# one_hot_a = utils.make_one_hot(torch_a, 3)
# print(one_hot_a)
# print(one_hot_a.shape)
#
# b = np.array([[[1, 2, 1],
#                [0, 1, 1],
#                [0, 0, 2]]])
# print(b.shape)
# print(b.dtype)
# torch_b = torch.from_numpy(b)
# print(torch_b.shape)
# print(torch_b.dtype)
# one_hot_b = utils.batch_make_one_hot(torch_b, 3)
# print(one_hot_b)
# print(one_hot_b.shape)
# print(one_hot_b.dtype)
#
# c = np.array([[[1, 2, 1],
#               [0, 1, 1],
#               [0, 0, 2]]])
# c = utils.ignore_background(c, 2, 1)
# mask = (c != 0)
# C = c[mask]
# print(mask)
# print(c)
# print(C)
#
# d = np.array([[[1, 2, 1],
#                [0, 1, 1],
#                [0, 0, 2]],
#               [[0, 0, 1],
#                [0, 1, 1],
#                [0, 0, 0]]
#               ])
# D = []
# D.extend(d[:, ])
# print(D)

# dataset preprocess experiments
# path1 = "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/ISIC/test/image/ISIC_0014749.jpg"
# path2 = "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/ISIC/test/mask/ISIC_0014749_segmentation.png"
# path3 = "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/valid/image/007-7210-400.jpg"
# path4 = "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/valid/annotation/007-7210-400.png"
#
# tensor_img1, tensor_target1 = preprocess.ISIC_preprocess(path1, path2, 256)[:2]
# tensor_img2, tensor_target2 = preprocess.DDR_preprocess(path3, path4, 512, 5)[:2]
# print(tensor_img1.shape)
# print(torch.max(tensor_img1))
# print(torch.min(tensor_img1))
# print(tensor_target1.shape)
# print(torch.max(tensor_target1))
# print(torch.min(tensor_target1))
# print(tensor_img2.shape)
# print(torch.max(tensor_img2))
# print(torch.min(tensor_img2))
# print(tensor_target2.shape)
# print(torch.max(tensor_target2))
# print(torch.min(tensor_target2))
#
# print("0:{}".format(torch.sum(torch.where(tensor_target2 == 0, 1, 0))))
# print("1:{}".format(torch.sum(torch.where(tensor_target2 == 1, 1, 0))))
# print("2:{}".format(torch.sum(torch.where(tensor_target2 == 2, 1, 0))))
# print("3:{}".format(torch.sum(torch.where(tensor_target2 == 3, 1, 0))))
# print("4:{}".format(torch.sum(torch.where(tensor_target2 == 4, 1, 0))))
#
# classes = ["EX", "HE", "MA", "SE"]
# pil_img1 = Image.open("/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/valid/annotation/007-7210-400.png")
# pil_img1 = utils.pil_crop_and_resize(pil_img1, crop_method="min", img_size=512, resize_method=Image.NEAREST)
# np_img1 = np.asarray(pil_img1)
# for i, c in enumerate(classes):
#     print("ann_{}:{}".format(c, np.sum(np.where(np_img1 == i + 1, 1, 0))))
#
#     pil_img2 = Image.open("/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/valid/label/{}/007-7210-400.tif".format(c))
#     pil_img2 = utils.pil_crop_and_resize(pil_img2, crop_method="min", img_size=512, resize_method=Image.NEAREST)
#     np_img2 = np.asarray(pil_img2)
#     print("ori_{}:{}".format(c, np.sum(np.where(np_img2 > 0, 1, 0))))
#
# a = [{"aa": np.nan}, {"aa": 1}, {"aa": None, "B": None}]
# b = [1, 2, np.nan, 2, 1.2]
# print(np.nan_to_num(a))
# print(np.nan_to_num(b))
#
# print(datasets.get_loss_weights("/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/train/annotation", 5, "png"))

class_weights = torch.FloatTensor([0.01, 1.])
loss_f0 = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
loss_f01 = nn.CrossEntropyLoss(weight=class_weights, reduction="sum")
loss_f1 = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")
# weight=class_weights
loss_f2 = CrossEntropyLoss(class_weight=[0.01, 1], reduction="none")
loss_f3 = CrossEntropyLoss(ignore_index=0, reduction="mean")
loss_f4 = CrossEntropyLoss(use_sigmoid=True, ignore_index=0, reduction="mean")
# class_weight=[0.01, 1.]

pred = np.array([[[[1, 1, 1],
                   [0, 0, 0],
                   [0, 0, 0]],
                  [[0, 0, 0],
                   [1, 1, 1],
                   [1, 1, 1]]
                  ],
                 [[[0, 0, 0],
                   [0, 1, 0],
                   [0, 1, 0]],
                  [[1, 1, 1],
                   [1, 0, 1],
                   [1, 0, 1]]
                  ]
                 ])
target = np.array([[[1, 0, 1],
                    [0, 1, 0],
                    [0, 0, 0]],
                   [[0, 1, 0],
                    [1, 0, 0],
                    [0, 1, 0]]]
                  )
print(pred.shape)
print(target.shape)
pred = torch.from_numpy(pred).float().requires_grad_(True)
target = torch.from_numpy(target).long()


# res0 = loss_f0(pred, target)
# print(res0)
# print(res0.sum())
# print(res0.mean())
# res01 = loss_f01(pred, target)
# print(res01)
# res1 = loss_f1(pred, target)
# print(res1)
# # print(res1.mean())
# res2 = loss_f2(pred, target)
# print(res2)
# # print(res2.mean())
# res3 = loss_f3(pred, target)
# print(res3)
#
# res4 = loss_f4(pred, target)
# print(res4)
#
# utils.get_dataset_norm(dataset_dir="/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/train/image",
#                        img_suffix="jpg", img_size=1024)

def cal_loss(loss_f, pred, target, weight=None):
    """
    :param pred:
    :param target:
    :param weight: list , int or float
    :return:
    """
    if len(loss_f) == 0:
        return None

    if weight is not None:
        assert isinstance(weight, list) and len(weight) == len(loss_f) or isinstance(weight, (int, float))
        if isinstance(weight, (int, float)):
            weight = [weight] * len(loss_f)
        weight = torch.as_tensor(weight)
        loss = loss_f[0](pred, target) * weight[0]
        for i, loss_func in enumerate(loss_f[1:], start=1):
            loss += loss_func(pred, target) * weight[i]
    else:
        loss = loss_f[0](pred, target)
        for i, loss_func in enumerate(loss_f[1:], start=1):
            loss += loss_func(pred, target)
    return loss


loss_F = [nn.CrossEntropyLoss(), smp.losses.DiceLoss(mode=smp.losses.constants.MULTICLASS_MODE)]
loss = 0
for loss_f in loss_F:
    loss += loss_f(pred, target)
loss1 = cal_loss(loss_F, pred, target, weight=[1, 0])
print(loss)
print(loss.backward())
print(loss)

print(loss1)
print(loss1.backward())
print(loss1)

# test mylogger
# a = logging.getLogger("sds")
# sh = logging.StreamHandler()
# log_formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
# sh.setLevel(logging.DEBUG)
# sh.setFormatter(log_formatter)
# a.addHandler(sh)
# a.setLevel(logging.DEBUG)
#
# A = Logger(None)
# AA = Logger(A)
#
# a.info("sds")
# A.info("sds")
# AA.info("sds")
