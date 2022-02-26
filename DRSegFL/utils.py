#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import codecs
import json
import pickle
import time

import yaml


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


def is_img(obj_type):
    img_type = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tga", ".svg", ".raw"}
    return obj_type in img_type


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
