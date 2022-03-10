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

from DRSegFL import utils, constants, preprocess


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


def label2annotation(img_path: str, label_paths: list, ann_path: str):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    shape = img.shape
    del img
    ann = np.zeros(shape, dtype=np.int32)

    for i, label_path in enumerate(label_paths):
        if not osp.exists(label_path):
            continue
        label = cv.imread(label_path, cv.IMREAD_GRAYSCALE)
        ann[label > 0] = i + 1
    ann_pil = Image.fromarray(ann.astype(np.uint8), mode="P")
    color_map = imgviz.label_colormap()
    ann_pil.putpalette(color_map)
    ann_pil.save(ann_path)
    return ann_pil


def labels2annotations(img_dir, target_dir, ann_dir, img_suffix, target_suffix, classes, dataset_type, custom_target_name=False, force=False):
    """
    Attention:img_name should be equal to target_name
    :param img_dir:.../img_files
    :param target_dir:.../class/target_files
    :param ann_dir:.../ann_files
    :param img_suffix:not include dot
    :param target_suffix:not include dot
    :param classes:the list of classes , not include "background(bg)"
    :param dataset_type:
    :param custom_target_name: optional custom, default target_name==img_name
    :param force: force to re-generate annotations
    """
    os.makedirs(ann_dir, exist_ok=True)
    for img_path in tqdm(glob.glob(osp.join(img_dir, "*.{}".format(img_suffix))), desc="{}_labels_to_annotations".format(dataset_type)):
        img_name = osp.basename(img_path).rsplit(".", 1)[0]
        target_name = img_name
        if custom_target_name:
            # Todo: custom this
            raise RuntimeError
        ann_path = osp.join(ann_dir, "{}.png".format(img_name))
        if not force and osp.exists(ann_path):
            continue
        label_paths = []
        for i, c in enumerate(classes):
            label_path = osp.join(target_dir, c, "{}.{}".format(target_name, target_suffix))
            label_paths.append(label_path)
        label2annotation(img_path, label_paths, ann_path)


def get_loss_weights(ann_dir, num_classes, ann_suffix):
    label_counts = [0] * num_classes
    files = glob.glob(osp.join(ann_dir, "*.{}".format(ann_suffix)))
    print(len(files))
    for file in files:
        img = Image.open(file)
        np_img = np.asarray(img)
        assert np.max(np_img) <= num_classes - 1, "values in [0,num_classes)"
        for i in range(num_classes):
            label_counts[i] += np.sum(np.where(np_img == i, 1, 0))
    avg_label_counts = np.array(label_counts) / len(files)
    print(avg_label_counts)
    loss_weights = sum(avg_label_counts) / avg_label_counts
    loss_weights /= np.min(loss_weights)
    return loss_weights


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
        img_path = self.datas_and_targets_path[index][0].strip()
        target_path = self.datas_and_targets_path[index][1].strip()

        # Todo: change dataset , modify below
        if self.dataset_name == constants.ISIC:
            return preprocess.ISIC_preprocess(img_path, target_path, self.img_size)[:2]
        elif self.dataset_name == constants.DDR:
            return preprocess.DDR_preprocess(img_path, target_path, self.img_size)[:2]

    def __len__(self):
        return len(self.datas_and_targets_path)


if __name__ == "__main__":
    print("compare PIL with cv2")
    pilimg1, img1 = utils.to_tensor_use_pil(Image.open("/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/ISIC/train/image/ISIC_0001126.jpg"),
                                            debug=True)
    cvimg1, img2 = utils.to_tensor_use_cv(cv.imread("/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/ISIC/train/image/ISIC_0001126.jpg"),
                                          debug=True)
    pilimg2, target1 = utils.to_tensor_use_pil(
        Image.open("/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/ISIC/train/mask/ISIC_0001126_segmentation.png").convert("L"), debug=True)
    cvimg2, target2 = utils.to_tensor_use_cv(
        cv.imread("/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/ISIC/train/mask/ISIC_0001126_segmentation.png", cv.IMREAD_GRAYSCALE),
        debug=True)
    print(torch.sum(abs(img1 - img2)))
    print(torch.sum(abs(target1 - target2)))

    print(np.asarray(pilimg1).shape)
    print(cvimg1.shape)
    print(np.asarray(pilimg2).shape)
    print(cvimg2.shape)
    print(np.sum(abs(np.asarray(pilimg1) - cvimg1)))
    print(np.sum(abs(np.asarray(pilimg2) - np.squeeze(cvimg2, 2))))
