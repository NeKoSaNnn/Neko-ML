#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import os.path as osp
import sys

sys.path.append(osp.dirname(sys.path[0]))
from neko import neko_utils


class utils(neko_utils.neko_utils):
    def __init__(self):
        super(utils, self).__init__("../log")
