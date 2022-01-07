#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

import datasets
from utils.utils import utils

utils = utils()


class LocalTrain(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        # 按指定idxs筛出dataset,并制作dataloader
        self.dataset_loader = DataLoader(datasets.SpiltDataSet(dataset, idxs),
                                         batch_size=self.args.local_bs, shuffle=True)
        self.loss_f = nn.CrossEntropyLoss()

    def train(self, net):
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        all_loss = []
        for ep in range(self.args.local_ep):
            for iter, (imgs, labels) in enumerate(self.dataset_loader):
                imgs, labels = imgs.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                res = net(imgs)
                loss = self.loss_f(res, imgs)
                loss.backward()
                optimizer.step()
                if self.args.verbose and iter % self.args.log_interval == 0:
                    utils.log(log_type="LocalTrain",
                              dict_val={"Epoch": ep, "Iter": iter, "Loss": format(loss.item(), ".4f")})
                all_loss.append(loss.item())
        return net.state_dict(), sum(all_loss) / len(all_loss)


class GlobalTrain():
    def __init__(self):
        pass
