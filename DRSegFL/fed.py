#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import numpy as np


def FedAvg(weights, contributions=None):
    # contributions 为clients的相应权重的贡献度
    # contributions is None 为完全平均值 否则为加权平均值
    # contributions 在这里用数据集的数量表征
    if contributions:
        assert len(weights) == len(contributions), "len(weights)!=len(contributions)"
    total_size = len(weights) if contributions is None else np.sum(contributions)
    avg_weights = [np.zeros(param.shape) for param in weights[0]]
    for c_index in range(len(weights)):
        for w_index in range(len(avg_weights)):
            avg_weights[w_index] += weights[c_index][w_index] / total_size if contributions is None \
                else weights[c_index][w_index] * contributions[c_index] / total_size
    return avg_weights
