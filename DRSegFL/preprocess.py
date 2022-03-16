#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import os.path as osp
import sys

import torch
from PIL import Image

root_dir_name = osp.dirname(sys.path[0])  # ...Neko-ML/
now_dir_name = sys.path[0]  # ...DRSegFL/
sys.path.append(root_dir_name)

from DRSegFL import utils, transforms as T
from torchvision import transforms


def ISIC_preprocess(img_path, target_path, img_size):
    """
    :param img_path:
    :param target_path:
    :param img_size:
    :return: tensor_img [Channel,H,W] ; tensor_target [1,H,W]:values in [0,1] ; pil_img ; pil_target
    """
    assert utils.is_img(target_path), "target must be img"
    pil_img = Image.open(img_path)
    pil_target = Image.open(target_path).convert("L")

    pil_trans = T.Compose([
        T.CenterCrop(max(pil_img.size)),
        T.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    tensor_trans = T.Compose([
        T.ToTensor(),
    ])

    pil_img, pil_target = pil_trans(pil_img, pil_target)

    tensor_img, tensor_target = tensor_trans(pil_img, pil_target)

    tensor_target[tensor_target > 0] = 1
    tensor_target = tensor_target.unsqueeze(0)
    return tensor_img, tensor_target, pil_img, pil_target


def DDR_preprocess(img_path: str, target_path: str, img_size: int, is_train: bool):
    """
    :param img_path: str
    :param target_path: str
    :param img_size: int
    :param is_train: bool
    :return: tensor_img [Channel,H,W] ; tensor_target [H,W]:values in [0,num_classes); pil_img ; pil_target
    """

    # foreground crop
    # fore_h, fore_w = utils.get_foreground_hw(img_path)
    # pad_h, pad_w = max(0, (fore_w - fore_h) // 2), max(0, (fore_h - fore_w) // 2)
    # trans = transforms.Compose([
    #     transforms.CenterCrop((fore_h, fore_w)),
    #     transforms.Pad((pad_w, pad_h)),
    #     transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC)
    # ])
    assert utils.is_img(target_path), "target must be img"
    pil_img = Image.open(img_path)
    pil_target = Image.open(target_path)
    if is_train:
        pil_trans = T.Compose([
            T.RandomCrop(img_size),
            # T.RandomResizedCrop(img_size, prob=0.5, interpolation=transforms.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
        ])
    else:
        pil_trans = T.Compose([
            # T.CenterCrop(min(pil_img.size)),
            T.Resize(img_size),
        ])

    tensor_trans = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.4211, 0.2640, 0.1104], std=[0.3133, 0.2094, 0.1256]),
    ])

    pil_img, pil_target = pil_trans(pil_img, pil_target)
    tensor_img, tensor_target = tensor_trans(pil_img, pil_target)

    return tensor_img, tensor_target, pil_img, pil_target


def DDR_OneLesion_preprocess(img_path: str, target_path: str, img_size: int, is_train: bool):
    """
    :param img_path: str
    :param target_path: str
    :param img_size: int
    :param is_train: bool
    :return: tensor_img [Channel,H,W] ; tensor_target [H,W]:values in [0,num_classes); pil_img ; pil_target
    """

    assert utils.is_img(target_path), "target must be img"
    pil_img = Image.open(img_path)
    pil_target = Image.open(target_path)
    if is_train:
        pil_trans = T.Compose([
            # T.RandomResizedCrop(img_size),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(),
        ])
    else:
        pil_trans = T.Compose([
            T.Resize(img_size)
        ])

    tensor_trans = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.4211, 0.2640, 0.1104], std=[0.3133, 0.2094, 0.1256]),
    ])

    pil_img, pil_target = pil_trans(pil_img, pil_target)
    tensor_img, tensor_target = tensor_trans(pil_img, pil_target)
    # tensor_target = tensor_target.unsqueeze(0)
    return tensor_img, tensor_target, pil_img, pil_target


if __name__ == "__main__":
    # image_path = "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/train/image/007-3399-200.jpg"
    # target_path = "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/train/annotation/007-3399-200.png"
    # timg, ttarget, pimg, ptarget = DDR_preprocess(image_path, target_path, 1024, is_train=False)
    # pimg.save("./tmp.jpg")
    # ptarget.save("./tmp.png")
    # print(torch.max(timg))
    # print(torch.min(timg))
    # print(torch.typename(timg))
    # print(torch.max(ttarget))
    # print(torch.min(ttarget))
    # print(torch.typename(ttarget))

    image_path = "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/EX/train/image/007-3399-200.jpg"
    target_path = "/home/maojingxin/workspace/Neko-ML/DRSegFL/datas/DDR_lesion_segmentation/EX/train/label/007-3399-200.png"
    timg, ttarget, pimg, ptarget = DDR_OneLesion_preprocess(image_path, target_path, 1024, is_train=False)
    pimg.save("./tmp.jpg")
    ptarget.save("./tmp.png")
    print(torch.max(timg))
    print(torch.min(timg))
    print(torch.typename(timg))
    print(torch.max(ttarget))
    print(torch.min(ttarget))
    print(torch.typename(ttarget))
