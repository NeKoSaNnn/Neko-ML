#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import copy

import torch


def FedAvg(local_w):
    # 以local_w[0]为基准
    avg_w = copy.deepcopy(local_w[0])
    for k in avg_w.keys():
        for i in range(1, len(local_w)):
            avg_w[k] += local_w[i][k]
        avg_w[k] = torch.div(avg_w[k], len(local_w))
    return avg_w
