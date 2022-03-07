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


def list_mean(list_, weight_list=None):
    if list_ is None:
        return None
    if weight_list:
        assert len(list_) == len(weight_list), "{}!={}".format(len(list_), len(weight_list))
    _len = sum(weight_list) if weight_list else len(list_)
    mean = 0
    for i, v in enumerate(list_):
        mean += (v * weight_list[i]) if weight_list else v
    mean /= _len
    return mean


def dict_list_mean(dict_list, weight_list=None):
    if dict_list is None:
        return None
    if weight_list:
        assert len(dict_list) == len(weight_list), "{}!={}".format(len(dict_list), len(weight_list))
    _len = sum(weight_list) if weight_list else len(dict_list)
    mean_dict = {}
    for k in dict_list[0].keys():
        mean_dict[k] = 0
        for i, d in enumerate(dict_list):
            mean_dict[k] += (d[k] * weight_list[i]) if weight_list else d[k]
        mean_dict[k] /= _len
    return mean_dict


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


def to_tensor_use_cv(cv_img, img_size=None, debug=False):
    # cv_img = cv.imread(img_path, cv.IMREAD_GRAYSCALE) if to_gray else cv.cvtColor(
    #     cv.imread(img_path, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
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


def to_tensor_use_pil(pil_img, img_size=None, debug=False):
    """
    :param pil_img:
    :param img_size:
    :param debug:
    :return: pil_img(after center-crop and resize),tensor_img(pil_img convert float(0.0-1.0) tensor)
    """
    if debug:
        print(pil_img.size)
        print(np.asarray(pil_img).shape)
    pil_img = transforms.CenterCrop(min(pil_img.size))(pil_img)  # crop
    if img_size:
        pil_img = pil_img.resize((img_size, img_size))
    tensor_img = transforms.ToTensor()(pil_img)
    if debug:
        print(tensor_img.shape)
    return pil_img, tensor_img


def to_label_use_pil(pil_img, img_size=None, debug=False):
    """
    :param pil_img:
    :param img_size:
    :param debug:
    :return: pil_img(after center-crop and resize),label_img(pil_img convert long tensor)
    """
    if debug:
        print(pil_img.size)
        print(np.asarray(pil_img).shape)
    pil_img = transforms.CenterCrop(min(pil_img.size))(pil_img)  # crop
    if img_size:
        pil_img = pil_img.resize((img_size, img_size))
    if debug:
        print(pil_img.size)
    label_img = torch.from_numpy(np.asarray(pil_img, dtype=np.long))
    return pil_img, label_img


def make_one_hot(label, num_classes, ignore=0):
    """
    :param label: [*], values in [0,num_classes]
    :param num_classes: C
    :param ignore: ignore value of background, here is 0
    :return: [C, *]
    """
    label = label.unsqueeze(0)
    shape = list(label.shape)
    shape[0] = num_classes + 1

    if ignore is not None:
        label[label == ignore] = num_classes + 1
        label = label - 1

    result = torch.zeros(shape, device=label.device)
    result.scatter_(0, label, 1)

    return result[:-1, ]


def batch_make_one_hot(labels, num_classes, ignore=0):
    """
    :param labels: [N, *], values in [0,num_classes)
    :param num_classes: C
    :param ignore: ignore value of background, here is 0
    :return: [N, C, *]
    """
    labels = labels.unsqueeze(1)
    shape = list(labels.shape)
    shape[1] = num_classes + 1

    if ignore is not None:
        labels[labels == ignore] = num_classes + 1
        labels = labels - 1

    result = torch.zeros(shape, device=labels.device)
    result.scatter_(1, labels, 1)

    return result[:, :-1, ]


def ignore_background(array, num_classes, ignore=0):
    """
    :param array: [*],values in [0,num_classes]
    :param num_classes: C
    :param ignore: ignore value of background, here is 0
    :return: [*] which ignore_index=num_classes
    """
    array[array == ignore] = num_classes + 1
    array[array > ignore] -= 1
    return array


def draw_predict(num_classes, img: Image.Image, target_mask: Image.Image, predict_mask, save_path):
    assert predict_mask.shape[0] == num_classes, "{}!={}".format(predict_mask.shape[0], num_classes)
    plt.figure(figsize=(20, 20))
    if num_classes > 1:
        plt.subplots_adjust(top=0.96, bottom=0.01, left=0.02, right=0.98, hspace=0.01, wspace=0.01)
        ax = plt.subplot(2, 2, 1)
        ax.set_title("ori img", fontsize=30)
        ax.imshow(img)
        ax.axis("off")
        ax = plt.subplot(2, 2, 2)
        ax.set_title("ground truth", fontsize=30)
        ax.imshow(target_mask)
        ax.axis("off")
        for i in range(num_classes):
            ax = plt.subplot(2, num_classes, num_classes + i + 1)
            ax.set_title("segmap(class {})".format(i + 1), fontsize=30)
            ax.imshow(predict_mask[i, :, :], cmap="gray")
            ax.axis("off")
    elif num_classes == 1:
        plt.subplots_adjust(top=0.96, bottom=0.01, left=0.02, right=0.98, hspace=0, wspace=0.01)
        ax = plt.subplot(1, 3, 1)
        ax.set_title("ori img", fontsize=30)
        ax.imshow(img)
        ax.axis("off")
        ax = plt.subplot(1, 3, 2)
        ax.set_title("ground truth", fontsize=30)
        ax.imshow(target_mask, cmap="gray")
        ax.axis("off")
        ax = plt.subplot(1, 3, 3)
        ax.set_title("segmap(class 1)", fontsize=30)
        ax.imshow(predict_mask[0, :, :], cmap="gray")
        ax.axis("off")
    else:
        raise AssertionError("num_classes >= 1")
    plt.xticks([]), plt.yticks([])
    plt.savefig(save_path)
    print("predict result saved to {}".format(save_path))
    plt.show()
