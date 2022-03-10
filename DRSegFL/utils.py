#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import codecs
import json
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
        mean += (np.nan_to_num(v) * weight_list[i]) if weight_list else np.nan_to_num(v)
    mean /= _len
    if torch.is_tensor(mean):
        mean = mean.item()
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
            mean_dict[k] += (np.nan_to_num(d[k]) * weight_list[i]) if weight_list else np.nan_to_num(d[k])
        mean_dict[k] /= _len
        if torch.is_tensor(mean_dict[k]):
            mean_dict[k] = mean_dict[k].item()
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


def get_foreground_hw(img_path):
    """
    according to the center symmetry, get the foreground h and w
    :param img_path:
    :return: foreground_h, foreground_w
    """
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    h, w = img.shape
    center_h, center_w = h // 2, w // 2
    mask = np.where(img > 0, 1, 0)
    i_h, i_w = mask.nonzero()
    min_h, max_h = np.min(i_h), np.max(i_h)
    min_w, max_w = np.min(i_w), np.max(i_w)
    foreground_h = 2 * max(abs(min_h - center_h), abs(max_h - center_h))
    foreground_w = 2 * max(abs(min_w - center_w), abs(max_w - center_w))
    return foreground_h, foreground_w


def pil_crop_and_resize(pil_img, crop_method="min", crop_size=None, img_size=None, resize_method=None, debug=False):
    if debug:
        print("ori")
        print(pil_img.size)
        print(np.asarray(pil_img).shape)
    if crop_method is None:  # no crop
        pass
    elif crop_method == "min":
        pil_img = transforms.CenterCrop(min(pil_img.size))(pil_img)  # crop
    elif crop_method == "max":
        pil_img = transforms.CenterCrop(max(pil_img.size))(pil_img)  # crop
    else:
        raise AssertionError("crop_method:{} error".format(crop_method))
    if debug:
        print("after crop")
        print(pil_img.size)
        print(np.asarray(pil_img).shape)
    if img_size:
        pil_img = pil_img.resize((img_size, img_size), resample=resize_method)
    if debug:
        print("after resize")
        print(pil_img.size)
        print(np.asarray(pil_img).shape)
    return pil_img


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


def to_tensor_use_pil(pil_img, crop_method="min", img_size=None, resize_method=None, debug=False):
    """
    :param pil_img:PIL.Image
    :param crop_method:"min","max",None
    :param img_size:int
    :param resize_method: PIL.Image.NEAREST,...,None
    :param debug:bool
    :return: pil_img(after center-crop and resize),tensor_img(pil_img convert float(0.0-1.0) tensor)
    """
    pil_img = pil_crop_and_resize(pil_img, crop_method=crop_method, img_size=img_size, resize_method=resize_method, debug=debug)
    tensor_img = transforms.ToTensor()(pil_img)
    if debug:
        print(tensor_img.shape)
    return pil_img, tensor_img


def to_label_use_pil(pil_img, crop_method="min", img_size=None, debug=False):
    """
    :param pil_img:PIL.Image
    :param crop_method:"min","max",None
    :param img_size:int , resize method is always PIL.Image.NEAREST
    :param debug:bool
    :return: pil_img(after center-crop and resize),label_img(pil_img convert long tensor)
    """

    pil_img = pil_crop_and_resize(pil_img, crop_method=crop_method, img_size=img_size, resize_method=Image.NEAREST, debug=debug)
    if debug:
        print(pil_img.size)
    label_img = torch.from_numpy(np.asarray(pil_img, dtype=np.long))
    return pil_img, label_img


def make_one_hot(label, num_classes):
    """
    :param label: [*], values in [0,num_classes)
    :param num_classes: C
    :return: [C, *]
    """
    label = label.unsqueeze(0)
    shape = list(label.shape)
    shape[0] = num_classes + 1

    result = torch.zeros(shape, device=label.device)
    result.scatter_(0, label, 1)

    return result[:-1, ]


def batch_make_one_hot(labels, num_classes):
    """
    :param labels: [N, *], values in [0,num_classes)
    :param num_classes: C
    :param ignore: ignore value of labels
    :return: [N, C, *]
    """
    labels = labels.unsqueeze(1)
    shape = list(labels.shape)
    shape[1] = num_classes + 1

    result = torch.zeros(shape, device=labels.device)
    result.scatter_(1, labels, 1)

    return result[:, :-1, ]


def ignore_background(array, num_classes, ignore=0):
    """
    :param array: [*],values in [0,num_classes)
    :param num_classes: C
    :param ignore: ignore value of background, here is 0
    :return: [*] which ignore_index=num_classes
    """
    array[array == ignore] = -1
    array[array > ignore] -= 1
    return array


def draw_predict(classes: list, img: Image.Image, target_mask: Image.Image, predict_mask, save_path, draw_gt_one_hot=False, ignore_index=-1):
    num_classes = len(classes)
    assert predict_mask.shape[0] == num_classes, "{}!={}".format(predict_mask.shape[0], num_classes)
    plt.figure(figsize=(25, 25))
    if num_classes > 1:
        flag = ignore_index in range(num_classes)
        plt.subplots_adjust(top=0.96, bottom=0, left=0.02, right=0.98, hspace=0.01, wspace=0.01)
        ax = plt.subplot(2 + draw_gt_one_hot, 2, 1)
        ax.set_title("ori img", fontsize=30)
        ax.imshow(img)
        ax.axis("off")
        ax = plt.subplot(2 + draw_gt_one_hot, 2, 2)
        ax.set_title("gt color_map", fontsize=30)
        ax.imshow(target_mask)
        ax.axis("off")
        for i in range(num_classes):
            if ignore_index != i:
                ax = plt.subplot(2 + draw_gt_one_hot, num_classes - flag, num_classes + i + 1 - flag - (flag and i > ignore_index))
                ax.set_title("segmap(class {})".format(i + 1) if classes is None else classes[i], fontsize=30)
                ax.imshow(predict_mask[i, :, :], cmap="gray")
                ax.axis("off")
        if draw_gt_one_hot:
            for i in range(num_classes):
                if ignore_index != i:
                    one_hot_gt = make_one_hot(torch.from_numpy(np.asarray(target_mask, dtype=np.long)), num_classes)
                    ax = plt.subplot(2 + draw_gt_one_hot, num_classes - flag, 2 * (num_classes - flag) + i + 1 - (flag and i > ignore_index))
                    ax.set_title("gt(class {})".format(i + 1) if classes is None else "gt({})".format(classes[i]), fontsize=30)
                    ax.imshow(one_hot_gt[i, :, :], cmap="gray")
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
        ax.set_title("segmap" if classes is None else classes[0], fontsize=30)
        ax.imshow(predict_mask[0, :, :], cmap="gray")
        ax.axis("off")
    else:
        raise AssertionError("num_classes({}) >= 1".format(num_classes))
    plt.xticks([]), plt.yticks([])
    plt.savefig(save_path)
    print("predict result saved to {}".format(save_path))
    plt.show()
