#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import json
import time


def load_json(f_name):
    with open(f_name) as f:
        return json.load(f)


def get_now_day():
    return time.strftime("%Y_%m_%d", time.localtime())


def is_img(obj_type):
    img_type = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tga", ".svg", ".raw"}
    return obj_type in img_type
