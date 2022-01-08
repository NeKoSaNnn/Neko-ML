#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.Dropout(),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
