#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
from os import path as osp

import cv2 as cv
import torch
from PIL import Image
from torchvision import transforms

from models import UNet
from utils.utils import utils
from matplotlib import pyplot as plt

from torch import nn

utils = utils(log_path="./log")


def test(model, test_object):
    model.eval()
    utils.mkdir_nf(save_path)
    with torch.no_grad():
        trans = transforms.Compose([transforms.Resize([256, 256]), transforms.ToTensor()])
        toPIL = transforms.ToPILImage()
        test_object = trans(test_object).unsqueeze(0)
        res = torch.squeeze(model(test_object))
        res = toPIL(res).convert("L")
        return res


if __name__ == "__main__":
    save_path = "./save/test"
    model = UNet.UNet(num_classes=1)
    model.load_state_dict(torch.load("./save/pt/Train-isic-unet-ep200-2022-01-13-00-04-38.pt"))
    plt.figure()

    test_img_path = "./data/ISIC/test/image/ISIC_0000102.jpg"
    test_mask_path = "./data/ISIC/test/mask/ISIC_0000102_segmentation.png"

    test_image = cv.imread(test_img_path)
    plt.subplot(131)
    plt.xlabel("ori image")
    plt.imshow(cv.resize(test_image, (256, 256), interpolation=cv.INTER_LINEAR))

    res = test(model, test_image)
    plt.subplot(132)
    plt.xlabel("result")
    plt.imshow(res)

    truth = cv.imread(test_mask_path)
    truth = cv.resize(truth, (256, 256), interpolation=cv.INTER_LINEAR)
    plt.subplot(133)
    plt.xlabel("ground truth")
    plt.imshow(truth)

    plt.savefig(osp.join(save_path, "test_res_" + utils.get_now_time() + ".png"))
