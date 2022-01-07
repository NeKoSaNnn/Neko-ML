#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(F.dropout(self.fc1(x), training=True))
        x = F.relu(F.dropout(self.fc2(x), training=True))

        # x = self.layer_input(x)
        # x = self.dropout(x)
        # x = self.relu(x)
        # x = self.layer_hidden(x)
        return x
