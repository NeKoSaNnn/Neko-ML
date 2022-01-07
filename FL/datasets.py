#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""

from torch.utils.data import Dataset


class SpiltDataSet(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


