#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import copy

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import fed
import datasets
from utils.utils import utils

utils = utils()


class LocalTrain(object):
    def __init__(self, args, dataloader):
        self.args = args
        # 按指定idxs筛出dataset,并制作dataloader
        self.dataset_loader = dataloader
        self.loss_f = nn.CrossEntropyLoss()

    def train(self, local_net):
        # local net
        local_net.train()
        optimizer = optim.SGD(local_net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        all_loss = []
        for ep in range(self.args.local_ep):
            for iter, (imgs, labels) in enumerate(self.dataset_loader):
                imgs, labels = imgs.to(self.args.device), labels.to(self.args.device)
                local_net.zero_grad()
                res = local_net(imgs)
                loss = self.loss_f(res, imgs)
                loss.backward()
                optimizer.step()
                if self.args.verbose and iter % self.args.log_interval == 0:
                    utils.log(log_type="LocalTrain",
                              dict_val={"Epoch": ep, "Iter": iter, "Loss": format(loss.item(), ".4f")})
                all_loss.append(loss.item())
        return local_net.state_dict(), np.array(all_loss).mean()


class GlobalTrain(object):
    def __init__(self, args):
        self.args = args
        self.initDataSet = datasets.InitDataSet(self.args)
        self.train_datasets, self.test_datasets = self.initDataSet.get()
        self.user_dataidx = self.initDataSet.get_iid_user_dataidx(self.train_datasets)

    def train(self, global_net):
        # global net
        global_net.train()
        global_w = global_net.state_dict()
        all_global_loss = []
        if self.args.all_clients:
            utils.divide_line("aggregation over all clients")
        for ep in range(self.args.epochs):
            # init all-local loss
            local_loss = []
            # init all-local w
            local_w = [global_w] * self.args.num_users if self.args.all_clients else []
            # args.epochs为全局epoch
            client_num = max(int(self.args.client_frac * self.args.num_users), 1)
            # 随机选取 client idx
            client_idxs = np.random.choice(range(self.args.num_users), client_num, replace=False)
            for c_id in client_idxs:
                local = LocalTrain(self.args,
                                   self.initDataSet.get_iid_dataloader(self.train_datasets, self.user_dataidx[c_id]))
                w, loss = local.train(copy.deepcopy(global_net).to(self.args.device))
                if self.args.all_clients:
                    local_w[c_id] = copy.deepcopy(w)
                else:
                    local_w.append(copy.deepcopy(w))
                local_loss.append(copy.deepcopy(loss))

            global_w = fed.FedAvg(local_w)
            global_net.load_state_dict(global_w)
            global_loss = np.arange(local_loss).mean()
            all_global_loss.append(global_loss)
            utils.log("Global", {"epoch": ep, "Loss": format(global_loss, ".4f")})
        return global_net
