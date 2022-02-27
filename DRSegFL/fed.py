#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import copy

import torch


def FedAvg(clients_weights, clients_contribution=None):
    # clients_contribution 为clients的相应权重的贡献度
    # clients_contribution is None 为完全平均值 否则为加权平均值
    # clients_contribution 在这里用数据集的数量表征
    if clients_contribution:
        assert len(clients_weights) == len(clients_contribution), "len(weights)!=len(contributions)"
    total_size = len(clients_weights) if clients_contribution is None else sum(clients_contribution)
    avg_weights = copy.deepcopy(clients_weights[0])

    for k in avg_weights.keys():
        if clients_contribution is not None:
            avg_weights[k] = torch.mul(avg_weights[k], clients_contribution[0])
        for i in range(1, len(clients_weights)):
            avg_weights[k] += clients_weights[i][k] if clients_contribution is None else torch.mul(
                clients_weights[i][k], clients_contribution[i])
        avg_weights[k] = torch.div(avg_weights[k], total_size)
    return avg_weights
