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

utils = utils(log_path="./log")


def test(model, test_object, save_path="./save/test"):
    utils.mkdir_nf(save_path)
    trans = transforms.ToTensor()
    test_object = trans(test_object).unsqueeze(0)
    res = model(test_object)
    print(res.shape)
    res = Image.fromarray(res.detach().numpy().squeeze(0).squeeze(0), mode="L")
    res.save(osp.join(save_path, "test_res_" + utils.get_now_time() + ".png"))


if __name__ == "__main__":
    model = UNet.UNet(1)
    model.load_state_dict(torch.load("./save/pt/isic-unet-ep10-2022-01-12-18-09-36.pt"))
    test_image = cv.imread("./data/ISIC/test/image/ISIC_0000102.jpg")
    test(model, test_image)
