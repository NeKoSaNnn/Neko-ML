#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import codecs
import json
import os
import pickle
import time

import cv2 as cv
import matplotlib
import numpy as np
import torch
import yaml
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torchvision import transforms


def load_json(f_path):
    with open(f_path, "r") as f:
        return json.load(f)


def load_yaml(f_path, encoding="UTF-8"):
    with open(f_path, "r", encoding=encoding, ) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_now_day():
    return time.strftime("%Y%m%d", time.localtime())


def get_now_daytime():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())


def get_now_time():
    return time.strftime("%H%M%S", time.localtime())


def is_img(path):
    img_types = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tga", ".svg", ".raw"}
    for img_type in img_types:
        if img_type in path:
            return True
    return False


def split_dataset_with_client_nums():
    pass


def obj2pickle(obj, file_path=None):
    if file_path is None:
        return codecs.encode(pickle.dumps(obj), "base64").decode()
    else:
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        return file_path


def pickle2obj(pickle_or_filepath):
    # filepath
    if ".pkl" in pickle_or_filepath:
        with open(pickle_or_filepath, "rb") as f:
            obj = pickle.load(f)
    # pickle file
    else:
        obj = pickle.loads(codecs.decode(pickle_or_filepath.encode(), "base64"))
    return obj


def save_weights(weights, path):
    torch.save(weights, path)


def to_tensor_use_cv(img_path, img_size=None, to_gray=False, debug=False):
    cv_img = cv.imread(img_path, cv.IMREAD_GRAYSCALE) if to_gray else cv.cvtColor(
        cv.imread(img_path, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
    if debug:
        print(cv_img.shape)
    if img_size:
        cv_img = cv.resize(cv_img, (img_size, img_size), interpolation=cv.INTER_LINEAR)
    if len(cv_img.shape) == 2:
        cv_img = np.expand_dims(cv_img, axis=2)

    if debug:
        print(cv_img.shape)
    # HWC to CHW
    np_img = cv_img.transpose((2, 0, 1))
    np_img = np_img / 255 if np.max(np_img) > 1 else np_img
    tensor_img = torch.from_numpy(np_img)
    if debug:
        print(tensor_img.shape)
    return cv_img, tensor_img


def to_tensor_use_pil(img_path, img_size=None, to_gray=False, debug=False):
    pil_img = Image.open(img_path).convert("L") if to_gray else Image.open(img_path)
    if debug:
        print(pil_img.size)
        print(np.asarray(pil_img).shape)
    tensor_img = transforms.ToTensor()(pil_img)
    if img_size:
        tensor_img = F.interpolate(tensor_img.unsqueeze(0), img_size, mode="nearest").squeeze(0)
    if debug:
        print(tensor_img.shape)
    return pil_img, tensor_img


def draw_predict(config, img: Image.Image, target_mask: Image.Image, predict_mask, save_path):
    classes = config["num_classes"]
    fig, ax = plt.subplots(1, classes + 2)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=0)
    ax[0].set_title("ori img")
    ax[0].imshow(img)
    ax[0].axis("off")
    ax[1].set_title("ground truth")
    ax[1].imshow(target_mask)
    ax[1].axis("off")
    # if classes > 1:
    for i in range(classes):
        ax[2 + i].set_title("output mask (class {})".format(i + 1))
        ax[2 + i].imshow(predict_mask[i, :, :], cmap="gray")
        ax[2 + i].axis("off")
    # else:
    #     ax[1].set_title("output mask")
    #     ax[1].imshow(predict_mask)
    #     ax[1].axis("off")
    plt.xticks([]), plt.yticks([])
    plt.savefig(save_path)
    plt.show()
