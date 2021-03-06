#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import copy
import glob
import logging
import os
import shutil

import cv2 as cv
import numpy as np
import imgviz
import torch
from PIL import Image
from torchvision import transforms
import os.path as osp

from tqdm import tqdm

import utils, preprocess, datasets, constants, transforms as T, metrics
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
# path3 = "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/val/image/007-7210-400.jpg"
# path4 = "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/val/annotation/007-7210-400.png"
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
# #
# print("0:{}".format(torch.sum(torch.where(tensor_target2 == 0, 1, 0))))
# print("1:{}".format(torch.sum(torch.where(tensor_target2 == 1, 1, 0))))
# print("2:{}".format(torch.sum(torch.where(tensor_target2 == 2, 1, 0))))
# print("3:{}".format(torch.sum(torch.where(tensor_target2 == 3, 1, 0))))
# print("4:{}".format(torch.sum(torch.where(tensor_target2 == 4, 1, 0))))
#
# classes = ["EX", "HE", "MA", "SE"]
# pil_img1 = Image.open("/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/val/annotation/007-7210-400.png")
# pil_img1 = utils.pil_crop_and_resize(pil_img1, crop_method="min", img_size=1024, resize_method=Image.BILINEAR)
# np_img1 = np.asarray(pil_img1)
# for i, c in enumerate(classes):
#     print("ann_{}:{}".format(c, np.sum(np.where(np_img1 == i + 1, 1, 0))))
#
#     pil_img2 = Image.open("/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/val/label/{}/007-7210-400.tif".format(c))
#     pil_img2 = utils.pil_crop_and_resize(pil_img2, crop_method="min", img_size=1024, resize_method=Image.BILINEAR)
#     np_img2 = np.asarray(pil_img2)
#     print("ori_{}:{}".format(c, np.sum(np.where(np_img2 > 0, 1, 0))))
#
# a = [{"aa": np.nan}, {"aa": 1}, {"aa": None, "B": None}]
# b = [1, 2, np.nan, 2, 1.2]
# print(np.nan_to_num(a))
# print(np.nan_to_num(b))
#
# print(datasets.get_loss_weights("/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/train/annotation", 5, "png"))
#
# class_weights = torch.FloatTensor([0.01, 1.])
# loss_f0 = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
# loss_f01 = nn.CrossEntropyLoss(weight=class_weights, reduction="sum")
# loss_f1 = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")
# # weight=class_weights
# loss_f2 = CrossEntropyLoss(class_weight=[0.01, 1], reduction="none")
# loss_f3 = CrossEntropyLoss(ignore_index=0, reduction="mean")
# loss_f4 = CrossEntropyLoss(use_sigmoid=True, ignore_index=0, reduction="mean")
# # class_weight=[0.01, 1.]
#
# pred = np.array([
#     [[1, 1, 1],
#      [0, 0, 0],
#      [0, 0, 0]],
#     [[0, 0, 0],
#      [0, 1, 0],
#      [0, 1, 0]],
# ])
# target = np.array([
#     [[1, 0, 1],
#      [0, 1, 0],
#      [0, 0, 0]],
#     [[0, 1, 0],
#      [1, 0, 0],
#      [0, 1, 0]]
# ])
# print(pred.shape)
# print(target.shape)
# # pred = torch.from_numpy(pred).float().requires_grad_(True)
# pred = torch.from_numpy(pred).long()
# target = torch.from_numpy(target).long()
# pred = pred.unsqueeze(1)
# target = target.unsqueeze(1)
# all_acc, accs, ious, dices, f_values = metrics.cal_metric(pred, target, 2)
# print(all_acc, accs, ious, dices, f_values)
# mAcc = np.nanmean(accs)
# mDice = np.nanmean(dices)
# mIoU = np.nanmean(ious)
# print(mIoU)
# print(mDice)
# print(mAcc)
# mDice = metrics.IoU2Dice(mIoU)
# print(mDice)
# pred = pred.unsqueeze(1)
# target = target.unsqueeze(1)
# dice = metrics.multi_dice_coeff(pred, target, epsilon=1)
# print(dice)

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

# utils.cal_dataset_norm(dataset_dir="/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/train/image",
#                        img_suffix="jpg", img_size=1024)
# utils.cal_dataset_norm(dataset_dir="/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/EX/train/image",
#                        img_suffix="jpg", img_size=None)
# utils.cal_dataset_norm(dataset_dir="/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/HE/train/image",
#                        img_suffix="jpg", img_size=None)
# utils.cal_dataset_norm(dataset_dir="/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/MA/train/image",
#                        img_suffix="jpg", img_size=None)
# utils.cal_dataset_norm(dataset_dir="/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/SE/train/image",
#                        img_suffix="jpg", img_size=None)

# def cal_loss(loss_f, pred, target, weight=None):
#     """
#     :param pred:
#     :param target:
#     :param weight: list , int or float
#     :return:
#     """
#     if len(loss_f) == 0:
#         return None
#
#     if weight is not None:
#         assert isinstance(weight, list) and len(weight) == len(loss_f) or isinstance(weight, (int, float))
#         if isinstance(weight, (int, float)):
#             weight = [weight] * len(loss_f)
#         weight = torch.as_tensor(weight)
#         loss = loss_f[0](pred, target) * weight[0]
#         for i, loss_func in enumerate(loss_f[1:], start=1):
#             loss += loss_func(pred, target) * weight[i]
#     else:
#         loss = loss_f[0](pred, target)
#         for i, loss_func in enumerate(loss_f[1:], start=1):
#             loss += loss_func(pred, target)
#     return loss
#
#
# loss_F = [nn.CrossEntropyLoss(), smp.losses.DiceLoss(mode=smp.losses.constants.MULTICLASS_MODE)]
# loss = 0
# for loss_f in loss_F:
#     loss += loss_f(pred, target)
# loss1 = cal_loss(loss_F, pred, target, weight=[1, 0])
# print(loss)
# print(loss.backward())
# print(loss)
#
# print(loss1)
# print(loss1.backward())
# print(loss1)

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


# dataset_dir = "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation"
#
# lesions = ["EX", "HE", "MA", "SE"]
# _types = [constants.TRAIN, constants.VALIDATION, constants.TEST]

# for lesion in lesions:
#     root_dir = osp.join(dataset_dir, lesion)
#     for _type in _types:
#         dir = osp.join(root_dir, _type)
#         img_paths = sorted(glob.glob(osp.join(dataset_dir, _type, "crop", "image", "*.jpg")))
#         label_paths = sorted(glob.glob(osp.join(dataset_dir, _type, "crop", "label", lesion, "*.tif")))
#         img_dir = osp.join(dir, "image")
#         label_dir = osp.join(dir, "label")
#         os.makedirs(img_dir, exist_ok=True)
#         os.makedirs(label_dir, exist_ok=True)
#         assert len(img_paths) == len(label_paths)
#
#         ignore = []
#         for i, label_path in tqdm(enumerate(label_paths), desc="label_{}_{}".format(lesion, _type)):
#             label = Image.open(label_path).convert("1")
#             np_label = np.asarray(label)
#             np_label = np.where(np_label > 0, 1, 0)
#             if np.sum(np_label) == 0:
#                 ignore.append(i)
#                 continue
#
#             label = cv.imread(label_path, cv.IMREAD_GRAYSCALE)
#             label[label > 0] = 1
#             pil_label = Image.fromarray(label.astype(np.uint8), mode="P")
#             pil_trans = T.Compose([
#                 # T.CenterCrop(min(pil_label.size)),
#                 T.Resize(1024, interpolation=transforms.InterpolationMode.BICUBIC)])
#             pil_label, _ = pil_trans(pil_label)
#             color_map = imgviz.label_colormap()
#             pil_label.putpalette(color_map)
#             pil_label.save(osp.join(label_dir, osp.basename(label_path).rsplit(".", 1)[0] + ".png"))
#
#         for i, img_path in tqdm(enumerate(img_paths), desc="image_{}_{}".format(lesion, _type)):
#             if i in ignore:
#                 continue
#             pil_img = Image.open(img_path)
#             pil_trans = T.Compose([
#                 # T.CenterCrop(min(pil_img.size)),
#                 T.Resize(1024, interpolation=transforms.InterpolationMode.BICUBIC)])
#             pil_img, _ = pil_trans(pil_img)
#             pil_img.save(osp.join(img_dir, osp.basename(img_path)))
#


# for lesion in lesions:
#     root_dir = osp.join(dataset_dir, lesion)
#     for _type in _types:
#         dir = osp.join(dataset_dir, lesion, _type)
#         img_paths = sorted(glob.glob(osp.join(dir, "image", "*.jpg")))
#         label_paths = sorted(glob.glob(osp.join(dir, "label", "*.png")))
#         print("{}_{}_image={}".format(lesion, _type, len(img_paths)))
#         print("{}_{}_label={}".format(lesion, _type, len(label_paths)))

# val merge to test
# dataset_dir = "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation"
# #
# lesions = ["EX", "HE", "MA", "SE"]
# _types = [constants.TRAIN, constants.VALIDATION, constants.TEST]
#
# test_dir = osp.join(dataset_dir, constants.TEST, "crop_resize")
# val_dir = osp.join(dataset_dir, constants.VALIDATION, "crop_resize")
#
# val_image_dir = osp.join(val_dir, "image")
# val_label_dir = osp.join(val_dir, "label")
# val_annotation_dir = osp.join(val_dir, "annotation")
# test_image_dir = osp.join(test_dir, "image")
# test_label_dir = osp.join(test_dir, "label")
# test_annotation_dir = osp.join(test_dir, "annotation")
#
# print(len(glob.glob(osp.join(test_image_dir, "*.jpg"))))
# print(len(glob.glob(osp.join(test_annotation_dir, "*.png"))))
#
# # for val_image in glob.glob(osp.join(val_image_dir, "*.jpg")):
# #     shutil.copy(val_image, test_image_dir)
# # for val_annotation in glob.glob(osp.join(val_annotation_dir, "*.png")):
# #     shutil.copy(val_annotation, test_annotation_dir)
#
# for lesion in lesions:
#     lesion_val_label_dir = osp.join(val_label_dir, lesion)
#     lesion_test_label_dir = osp.join(test_label_dir, lesion)
#
#     print(len(glob.glob(osp.join(lesion_test_label_dir, "*.tif"))))
#     # for val_label in glob.glob(osp.join(lesion_val_label_dir, "*.tif")):
#     #     shutil.copy(val_label, lesion_test_label_dir)
#     print(len(glob.glob(osp.join(lesion_test_label_dir, "*.tif"))))
#
# print(len(glob.glob(osp.join(test_image_dir, "*.jpg"))))
# print(len(glob.glob(osp.join(test_annotation_dir, "*.png"))))

# dataset_dir = "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation"
#
# lesions = ["EX", "HE", "MA", "SE"]
# _types = [constants.TRAIN, constants.VALIDATION, constants.TEST]
#
# for lesion in lesions:
#     for _type in _types:
#         contour_dir = osp.join(dataset_dir, _type, "contour")
#         crop_dir = osp.join(dataset_dir, _type, "crop")
#         crop_resize_dir = osp.join(dataset_dir, _type, "crop_resize")
#
#         img_paths = sorted(glob.glob(osp.join(dataset_dir, _type, "image", "*.jpg")))
#         label_paths = sorted(glob.glob(osp.join(dataset_dir, _type, "label", lesion, "*.tif")))
#         annotation_paths = sorted(glob.glob(osp.join(dataset_dir, _type, "annotation", "*.png")))
#
#         assert len(img_paths) == len(label_paths) == len(annotation_paths)
#
#         contour_img_dir = osp.join(contour_dir, "image")
#         contour_label_dir = osp.join(contour_dir, "label", lesion)
#         contour_annotation_dir = osp.join(contour_dir, "annotation")
#
#         crop_img_dir = osp.join(crop_dir, "image")
#         crop_label_dir = osp.join(crop_dir, "label", lesion)
#         crop_annotation_dir = osp.join(crop_dir, "annotation")
#
#         crop_resize_img_dir = osp.join(crop_resize_dir, "image")
#         crop_resize_label_dir = osp.join(crop_resize_dir, "label", lesion)
#         crop_resize_annotation_dir = osp.join(crop_resize_dir, "annotation")
#
#         os.makedirs(contour_img_dir, exist_ok=True)
#         os.makedirs(contour_label_dir, exist_ok=True)
#         os.makedirs(contour_annotation_dir, exist_ok=True)
#
#         os.makedirs(crop_img_dir, exist_ok=True)
#         os.makedirs(crop_label_dir, exist_ok=True)
#         os.makedirs(crop_annotation_dir, exist_ok=True)
#
#         os.makedirs(crop_resize_img_dir, exist_ok=True)
#         os.makedirs(crop_resize_label_dir, exist_ok=True)
#         os.makedirs(crop_resize_annotation_dir, exist_ok=True)
#
#         for i, img_path in tqdm(enumerate(img_paths), desc="{}_{}".format(_type, lesion)):
#             contour_img_path = osp.join(contour_img_dir, osp.basename(img_path))
#             crop_img_path = osp.join(crop_img_dir, osp.basename(img_path))
#             crop_resize_img_path = osp.join(crop_resize_img_dir, osp.basename(img_path))
#
#             size, contours = utils.getContours(img_path, contour_img_path, 30)
#
#             crop_trans = T.CenterCrop(size)
#             resize_trans = T.Resize(1024, interpolation=transforms.InterpolationMode.BICUBIC)
#
#             label_path = label_paths[i]
#             annotation_path = annotation_paths[i]
#
#             contour_label_path = osp.join(contour_label_dir, osp.basename(label_path))
#             contour_annotation_path = osp.join(contour_annotation_dir, osp.basename(annotation_path))
#
#             crop_label_path = osp.join(crop_label_dir, osp.basename(label_path))
#             crop_annotation_path = osp.join(crop_annotation_dir, osp.basename(annotation_path))
#
#             crop_resize_label_path = osp.join(crop_resize_label_dir, osp.basename(label_path))
#             crop_resize_annotation_path = osp.join(crop_resize_annotation_dir, osp.basename(annotation_path))
#
#             label = cv.imread(label_path)
#             cv.drawContours(label, contours, -1, (0, 0, 255), 3)
#             cv.imwrite(contour_label_path, label)
#
#             annotation = cv.imread(annotation_path)
#             cv.drawContours(annotation, contours, -1, (0, 0, 255), 3)
#             cv.imwrite(contour_annotation_path, annotation)
#
#             pil_img = Image.open(img_path)
#             pil_label = Image.open(label_path)
#             pil_annotation = Image.open(annotation_path)
#
#             crop_img, crop_label = crop_trans(pil_img, pil_label)
#             crop_annotation, _ = crop_trans(pil_annotation)
#
#             crop_resize_img, crop_resize_label = resize_trans(crop_img, crop_label)
#             crop_resize_annotation, _ = resize_trans(crop_annotation)
#
#             crop_img.save(crop_img_path)
#             crop_label.save(crop_label_path)
#             crop_annotation.save(crop_annotation_path)
#
#             crop_resize_img.save(crop_resize_img_path)
#             crop_resize_label.save(crop_resize_label_path)
#             crop_resize_annotation.save(crop_resize_annotation_path)

# "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/train/image/007-1831-100.jpg"
# "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/train/image/007-1889-100.jpg"
# size, _ = utils.getContours("/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/train/image/007-1889-100.jpg",
#                             "./tmp2.jpg")
# print(size)

fgadr_path = "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/FGARD"
fgadr_img_dir = osp.join(fgadr_path, "images")
img = sorted(glob.glob(osp.join(fgadr_img_dir, "*.png")))

# for _lesion in ["HE", "EX", "SE"]:
#     labels = sorted(glob.glob(osp.join(fgadr_path, _lesion, "backup", "*.png")))
#     num_labels = len(labels)
#     train_dir = osp.join(fgadr_path, _lesion, "train")
#     test_dir = osp.join(fgadr_path, _lesion, "test")
#     for i, label in enumerate(labels):
#         if i < int(num_labels * 0.8):
#             shutil.copy(label, train_dir)
#         else:
#             shutil.copy(label, test_dir)
#     print("{}:{}".format(_lesion, num_labels))
#     print("{}_train:{}".format(_lesion, len(os.listdir(train_dir))))
#     print("{}_test:{}".format(_lesion, len(os.listdir(test_dir))))

# for _lesion in ["HE", "EX", "SE"]:
#     train_dir = osp.join(fgadr_path, _lesion, "train")
#     test_dir = osp.join(fgadr_path, _lesion, "test")
#     os.makedirs(osp.join(train_dir, "image"), exist_ok=True)
#     os.makedirs(osp.join(train_dir, "label"), exist_ok=True)
#     os.makedirs(osp.join(test_dir, "image"), exist_ok=True)
#     os.makedirs(osp.join(test_dir, "label"), exist_ok=True)
#     for label in glob.glob(osp.join(train_dir, "label", "*.png")):
#         name = osp.basename(label)
#         if osp.exists(osp.join(fgadr_img_dir, name)):
#             shutil.copy(osp.join(fgadr_img_dir, name), osp.join(train_dir, "image"))
#         else:
#             print(osp.join(fgadr_img_dir, name))
#     print(len(os.listdir(osp.join(train_dir, "image"))))
#     print(len(os.listdir(osp.join(train_dir, "label"))))
#     for label in glob.glob(osp.join(test_dir, "label", "*.png")):
#         name = osp.basename(label)
#         if osp.exists(osp.join(fgadr_img_dir, name)):
#             shutil.copy(osp.join(fgadr_img_dir, name), osp.join(test_dir, "image"))
#         else:
#             print(osp.join(fgadr_img_dir, name))
#     print(len(os.listdir(osp.join(test_dir, "image"))))
#     print(len(os.listdir(osp.join(test_dir, "label"))))

# utils.cal_dataset_norm("/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/FGARD/images", "png", 1024)

# for _lesion in ["HE", "EX", "SE"]:
#     for _type in ["train", "test"]:
#         for label_path in glob.glob(osp.join(fgadr_path, _lesion, _type, "label", "*.png")):
#             label = cv.imread(label_path, cv.IMREAD_GRAYSCALE)
#             label[label > 0] = 1
#             pil_label = Image.fromarray(label.astype(np.uint8), mode="P")
#             color_map = imgviz.label_colormap()
#             pil_label.putpalette(color_map)
#             pil_label.save(label_path)


for _lesion in ["HE", "EX", "SE"]:
    for _type in ["train", "test"]:
        resize_img_path = osp.join(fgadr_path, _lesion, "resize", _type, "image")
        resize_label_path = osp.join(fgadr_path, _lesion, "resize", _type, "label")
        os.makedirs(resize_img_path, exist_ok=True)
        os.makedirs(resize_label_path, exist_ok=True)
        for img_path in tqdm(glob.glob(osp.join(fgadr_path, _lesion, _type, "image", "*.png"))):
            img = Image.open(img_path)
            img = img.resize((1024, 1024), resample=Image.BILINEAR)
            img.save(osp.join(resize_img_path, osp.basename(img_path)))
        for label_path in tqdm(glob.glob(osp.join(fgadr_path, _lesion, _type, "label", "*.png"))):
            label = Image.open(label_path)
            label = label.resize((1024, 1024), resample=Image.BILINEAR)
            label.save(osp.join(resize_label_path, osp.basename(label_path)))

# img1 = cv.imread(list(HE)[222])
# print(np.max(img1))
# print(np.min(img1))
