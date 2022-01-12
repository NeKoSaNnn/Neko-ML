#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import copy

import numpy as np
from torch import nn, optim

import fed
from eval import Eval
from utils.utils import utils

utils = utils(log_path="./log")


class Train(object):
    def __init__(self, args, initDataSet):
        self.args = args
        self.initDataSet = initDataSet
        self.train_dataset, _, _ = self.initDataSet.get()
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.initDataSet.get_dataloader()
        # init eval
        self.train_eval, self.val_eval, self.test_eval = Eval(self.args, self.train_dataloader, utils, "Train"), \
                                                         Eval(self.args, self.val_dataloader, utils, "Val"), \
                                                         Eval(self.args, self.test_dataloader, utils, "Test")

    def train(self, net, is_eval=True, loss_f=nn.CrossEntropyLoss()):
        # train
        net.train()

        # optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer = optim.Adam(net.parameters())

        for ep in range(1, self.args.epochs + 1):
            net.train()
            iter_loss = .0
            tmp_iter_loss = .0
            for iter, (imgs, targets) in enumerate(self.train_dataloader, start=1):
                imgs, targets = imgs.to(self.args.device), targets.to(self.args.device)
                optimizer.zero_grad()
                res = net(imgs)
                loss = loss_f(res, targets)
                loss.backward()
                optimizer.step()
                iter_loss += loss.item()
                tmp_iter_loss += loss.item()
                if self.args.verbose and iter % self.args.log_interval == 0:
                    utils.log(log_type="Train",
                              dict_val={"Epoch": ep, "Iter": iter,
                                        "Loss": format(tmp_iter_loss / self.args.log_interval, ".4f")})
                    tmp_iter_loss = .0

            utils.log("Non_Fed", {"epoch": ep, "Loss": format(iter_loss / len(self.train_dataloader), ".4f")})

            # eval
            if is_eval and (ep % self.args.eval_interval == 0 or ep == self.args.epochs):
                # eval train
                self.train_eval.eval(net, get_best=True)
                # eval val
                self.val_eval.eval(net, get_best=True)
                # eval test
                self.test_eval.eval(net, get_best=True)

        return self.train_eval, self.val_eval, self.test_eval


class GlobalTrain(object):
    def __init__(self, args, initDataSet):
        assert args.iid
        self.args = args
        self.initDataSet = initDataSet
        self.train_dataset, _, _ = self.initDataSet.get()
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.initDataSet.get_dataloader()
        self.user_dataidx = self.initDataSet.get_iid_user_dataidx(self.train_dataset)
        # init eval
        self.train_eval, self.val_eval, self.test_eval = Eval(self.args, self.train_dataloader, utils, "Train"), \
                                                         Eval(self.args, self.val_dataloader, utils, "Val"), \
                                                         Eval(self.args, self.test_dataloader, utils, "Test")

    def train(self, global_net, is_eval=True, local_loss_f=nn.CrossEntropyLoss()):
        # global net
        global_net.train()

        global_w = global_net.state_dict()
        # init all-local w
        local_w = [global_w] * self.args.num_users if self.args.all_clients else []
        for ep in range(1, self.args.epochs + 1):
            global_net.train()
            # init all-local loss
            all_local_loss = .0
            # re-init all-local w for non-all clients
            if not self.args.all_clients:
                local_w = []
            # args.epochs为全局epoch
            client_num = max(int(self.args.client_frac * self.args.num_users), 1)
            # 随机选取 client idx
            client_idxs = np.random.choice(range(self.args.num_users), client_num, replace=False)
            for c_id in client_idxs:
                local = LocalTrain(self.args,
                                   self.initDataSet.get_iid_dataloader(self.train_dataset, self.user_dataidx[c_id]))
                w, loss = local.train(copy.deepcopy(global_net).to(self.args.device), loss_f=local_loss_f)
                if self.args.all_clients:
                    local_w[c_id] = copy.deepcopy(w)
                else:
                    local_w.append(copy.deepcopy(w))
                all_local_loss += copy.deepcopy(loss)

            global_loss = all_local_loss / client_num
            global_w = fed.FedAvg(local_w)
            global_net.load_state_dict(global_w)

            utils.log("Fed I.I.D Global", {"epoch": ep, "Loss": format(global_loss, ".4f")})

            # eval
            if is_eval and (ep % self.args.eval_interval == 0 or ep == self.args.epochs):
                # eval train
                self.train_eval.eval(global_net, get_best=True)
                # eval val
                self.val_eval.eval(global_net, get_best=True)
                # eval test
                self.test_eval.eval(global_net, get_best=True)

        return self.train_eval, self.val_eval, self.test_eval


class LocalTrain(object):
    def __init__(self, args, train_dataloader):
        # 按user分数据，制作dataloader
        self.args = args
        self.train_dataloader = train_dataloader

    def train(self, local_net, loss_f=nn.CrossEntropyLoss()):
        # local net
        local_net.train()

        # optimizer = optim.SGD(local_net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer = optim.Adam(local_net.parameters(), lr=self.args.lr)

        ep_loss = .0
        for ep in range(1, self.args.local_ep + 1):
            iter_loss = .0
            tmp_iter_loss = .0
            for iter, (imgs, targets) in enumerate(self.train_dataloader, start=1):
                imgs, targets = imgs.to(self.args.device), targets.to(self.args.device)
                optimizer.zero_grad()
                res = local_net(imgs)
                loss = loss_f(res, targets)
                loss.backward()
                optimizer.step()
                iter_loss += loss.item()
                tmp_iter_loss += loss.item()
                if self.args.verbose and iter % self.args.log_interval == 0:
                    utils.log(log_type="LocalTrain",
                              dict_val={"Epoch": ep, "Iter": iter,
                                        "Loss": format(tmp_iter_loss / self.args.log_interval, ".4f")})
                    tmp_iter_loss = .0
            ep_loss += iter_loss / len(self.train_dataloader)
        return local_net.state_dict(), ep_loss / self.args.local_ep
