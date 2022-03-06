#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""

import glob
import os
import os.path as osp

import cv2 as cv
import imgviz
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from DRSegFL import utils, constants


def dataset_txt_generate(imgs_dir: str, img_suffix: str, targets_dir: str, target_suffix: str, txt_file_path: str, is_augment: bool):
    """
    imgs in imgs_dir and targets in targets_dir
    :param imgs_dir:
    :param img_suffix:
    :param targets_dir:
    :param target_suffix:
    :param txt_file_path:
    :param is_augment:
    :return:
    """
    if is_augment:
        # .../augment/...
        imgs = sorted(glob.glob(osp.join(imgs_dir, "**/*.{}".format(img_suffix)), recursive=True))
        targets = sorted(glob.glob(osp.join(targets_dir, "**/*.{}".format(target_suffix)), recursive=True))
    else:
        imgs = sorted(glob.glob(osp.join(imgs_dir, "*.{}".format(img_suffix))))
        targets = sorted(glob.glob(osp.join(targets_dir, "*.{}".format(target_suffix))))
    assert len(imgs) == len(targets), "len(datas):{} != len(targets):{}".format(len(imgs), len(targets))
    lines = [imgs[i] + " " + targets[i] + "\n" for i in range(len(imgs))]

    with open(txt_file_path, "w+") as f:
        f.writelines(lines)


def iid_dataset_txt_generate(imgs_dir: str, img_suffix: str, targets_dir: str, target_suffix: str, txt_file_paths: list, is_augment: bool):
    """
    imgs in imgs_dir and targets in targets_dir
    :param imgs_dir:
    :param img_suffix:
    :param targets_dir:
    :param target_suffix:
    :param txt_file_paths:
    :param is_augment:
    :return:
    """
    if is_augment:
        # .../augment/...
        imgs = sorted(glob.glob(osp.join(imgs_dir, "**/*.{}".format(img_suffix)), recursive=True))
        targets = sorted(glob.glob(osp.join(targets_dir, "**/*.{}".format(target_suffix)), recursive=True))
    else:
        imgs = sorted(glob.glob(osp.join(imgs_dir, "*.{}".format(img_suffix))))
        targets = sorted(glob.glob(osp.join(targets_dir, "*.{}".format(target_suffix))))
    assert len(imgs) == len(targets), "len(datas):{} != len(targets):{}".format(len(imgs), len(targets))
    lines = [imgs[i] + " " + targets[i] + "\n" for i in range(len(imgs))]

    split_num = len(txt_file_paths)
    per_split_data_size = len(lines) // split_num
    for i in range(split_num - 1):
        now_split_lines = np.random.choice(lines, per_split_data_size, replace=False)
        lines = list(set(lines) - set(now_split_lines))
        with open(txt_file_paths[i], "w+") as f:
            f.writelines(now_split_lines)
    with open(txt_file_paths[-1], "w+") as f:
        f.writelines(lines)


def labels2annotations(img_dir, target_dir, ann_dir, img_suffix, target_suffix, classes, dataset_type):
    """
    Attention:img_name should be equal to target_name
    :param img_dir:
    :param target_dir:
    :param ann_dir:
    :param img_suffix:not include dot
    :param target_suffix:not include dot
    :param classes:the list of classes
    :param dataset_type:
    """
    os.makedirs(ann_dir, exist_ok=True)
    for file_path in tqdm(glob.glob(osp.join(img_dir, "*.{}".format(img_suffix))), desc="{}_labels_to_annotations".format(dataset_type)):
        file_name = osp.basename(file_path).rsplit(".", 1)[0]
        img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        shape = img.shape
        ann = np.zeros(shape, dtype=np.int32)
        ann_path = osp.join(ann_dir, "{}.png".format(file_name))

        for i, c in enumerate(classes):
            label_path = osp.join(target_dir, c, "{}.{}".format(file_name, target_suffix))
            if not osp.exists(label_path):
                continue
            label = cv.imread(label_path, cv.IMREAD_GRAYSCALE)
            ann[label > 0] = i + 1
        ann_pil = Image.fromarray(ann.astype(np.uint8), mode="P")
        color_map = imgviz.label_colormap()
        ann_pil.putpalette(color_map)
        ann_pil.save(ann_path)


def dataset_augment(img_dir, target_dir, img_suffix, target_suffix, dataset_type):
    imgs = sorted(glob.glob(osp.join(img_dir, "*.{}".format(img_suffix))))
    targets = sorted(glob.glob(osp.join(target_dir, "*.{}".format(target_suffix))))
    augment_img_dir = osp.join(img_dir, "augment")
    os.makedirs(augment_img_dir, exist_ok=True)
    augment_target_dir = osp.join(target_dir, "augment")
    os.makedirs(augment_target_dir, exist_ok=True)
    assert len(imgs) == len(targets), "len(imgs)!=len(targets)"
    for img_path in tqdm(imgs, desc="{}_imgs_augment".format(dataset_type)):
        img_name = osp.basename(img_path).strip().rsplit(".", 1)[0]
        img_pil = Image.open(img_path)
        img_pil.transpose(Image.ROTATE_90).save(osp.join(augment_img_dir, "{}_90.{}".format(img_name, img_suffix)))
        img_pil.transpose(Image.ROTATE_180).save(osp.join(augment_img_dir, "{}_180.{}".format(img_name, img_suffix)))
        img_pil.transpose(Image.ROTATE_270).save(osp.join(augment_img_dir, "{}_270.{}".format(img_name, img_suffix)))
        img_pil.transpose(Image.FLIP_LEFT_RIGHT).save(osp.join(augment_img_dir, "{}_horizontal.{}".format(img_name, img_suffix)))
        img_pil.transpose(Image.FLIP_TOP_BOTTOM).save(osp.join(augment_img_dir, "{}_vertical.{}".format(img_name, img_suffix)))

    for target_path in tqdm(targets, desc="{}_targets_augment".format(dataset_type)):
        target_name = osp.basename(target_path).strip().rsplit(".", 1)[0]
        target_pil = Image.open(target_path)
        target_pil.transpose(Image.ROTATE_90).save(osp.join(augment_target_dir, "{}_90.{}".format(target_name, target_suffix)))
        target_pil.transpose(Image.ROTATE_180).save(osp.join(augment_target_dir, "{}_180.{}".format(target_name, target_suffix)))
        target_pil.transpose(Image.ROTATE_270).save(osp.join(augment_target_dir, "{}_270.{}".format(target_name, target_suffix)))
        target_pil.transpose(Image.FLIP_LEFT_RIGHT).save(osp.join(augment_target_dir, "{}_horizontal.{}".format(target_name, target_suffix)))
        target_pil.transpose(Image.FLIP_TOP_BOTTOM).save(osp.join(augment_target_dir, "{}_vertical.{}".format(target_name, target_suffix)))


class ListDataset(Dataset):
    def __init__(self, txt_path: str, dataset_name: str, num_classes: int, img_size=256, ):
        with open(txt_path, "r") as f:
            self.datas_and_targets_path = f.readlines()
        self.datas_and_targets_path = [data_and_target.strip().split(" ", 1) for data_and_target in
                                       self.datas_and_targets_path]
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.img_size = img_size

    def __getitem__(self, index):
        # Todo: change dataset , modify below
        if self.dataset_name == constants.ISIC:
            return self._ISIC_getitem(index)
        elif self.dataset_name == constants.DDR:
            return self._DDR_getitem(index)

    def __len__(self):
        return len(self.datas_and_targets_path)

    def _ISIC_getitem(self, index):
        img_path = self.datas_and_targets_path[index][0].strip()
        target_path = self.datas_and_targets_path[index][1].strip()

        _, tensor_img = utils.to_tensor_use_pil(img_path, self.img_size)

        if utils.is_img(target_path):
            _, tensor_target = utils.to_tensor_use_pil(target_path, self.img_size, to_gray=True)
        else:
            raise InterruptedError("标签数据非图片数据，需要额外处理")
        return tensor_img, tensor_target, img_path, target_path

    def _DDR_getitem(self, index):
        img_path = self.datas_and_targets_path[index][0].strip()
        target_path = self.datas_and_targets_path[index][1].strip()

        _, tensor_img = utils.to_tensor_use_pil(img_path, self.img_size)

        if utils.is_img(target_path):
            target = Image.open(target_path).resize((self.img_size, self.img_size))
            tensor_target = torch.from_numpy(np.asarray(target, dtype=np.long))
            tensor_target[tensor_target == 0] = 256
            tensor_target = tensor_target - 1
            # tensor_target = utils.make_one_hot(tensor_target, self.num_classes)
        else:
            raise InterruptedError("标签数据非图片数据，需要额外处理")
        return tensor_img, tensor_target, img_path, target_path


if __name__ == "__main__":
    print("compare PIL with cv2")
    pilimg1, img1 = utils.to_tensor_use_pil("/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/ISIC/train/image/ISIC_0001126.jpg",
                                            debug=True)
    cvimg1, img2 = utils.to_tensor_use_cv("/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/ISIC/train/image/ISIC_0001126.jpg", debug=True)
    pilimg2, target1 = utils.to_tensor_use_pil(
        "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/ISIC/train/mask/ISIC_0001126_segmentation.png",
        to_gray=True, debug=True)
    cvimg2, target2 = utils.to_tensor_use_cv(
        "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/ISIC/train/mask/ISIC_0001126_segmentation.png",
        to_gray=True, debug=True)
    print(torch.sum(abs(img1 - img2)))
    print(torch.sum(abs(target1 - target2)))

    print(np.asarray(pilimg1).shape)
    print(cvimg1.shape)
    print(np.asarray(pilimg2).shape)
    print(cvimg2.shape)
    print(np.sum(abs(np.asarray(pilimg1) - cvimg1)))
    print(np.sum(abs(np.asarray(pilimg2) - np.squeeze(cvimg2, 2))))
