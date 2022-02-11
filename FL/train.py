#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import copy

import numpy as np
import torch
from torch import nn, optim

import fed
from eval import Eval
from utils.utils import utils
from torch.nn import functional as F
from metrics import dice_loss

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

    def trainMultiClassifier(self, net, loss_f, is_eval=True):
        # train
        net.train()

        optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer = optim.Adam(net.parameters())
        # optimizer = optim.RMSprop(net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay,
        #                           momentum=self.args.momentum)

        lr_schedule = optim.lr_scheduler.StepLR(optimizer, 20, 0.1)  # 每20个epoch lr=lr*0.1

        for ep in range(1, self.args.epochs + 1):
            net.train()
            iter_loss = 0
            tmp_iter_loss = 0
            for iter, (imgs, targets) in enumerate(self.train_dataloader, start=1):
                assert imgs.shape[1] == self.args.num_channels
                imgs, targets = imgs.to(self.args.device), targets.to(self.args.device)

                preds = net(imgs)
                loss = loss_f(preds, targets)
                # + dice_loss(F.softmax(res, dim=1).float(),F.one_hot(targets, self.args.num_classes).permute(0, 3, 1, 2).float(), multiclass=True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_loss += loss.item()
                tmp_iter_loss += loss.item()
                if self.args.verbose and iter % self.args.log_interval == 0:
                    utils.log(log_type="Train",
                              dict_val={"Epoch": ep, "Iter": iter,
                                        "Loss": format(tmp_iter_loss / self.args.log_interval, ".4f")})
                    tmp_iter_loss = .0
            lr_schedule.step()
            utils.log("Non_Fed", {"Epoch": ep, "Avg_Loss": format(iter_loss / len(self.train_dataloader), ".4f")})

            # eval
            if is_eval and (ep % self.args.eval_interval == 0 or ep == self.args.epochs):
                # eval train
                self.train_eval.eval(net, get_best=True)
                # eval val
                self.val_eval.eval(net, get_best=True)
                # eval test
                self.test_eval.eval(net, get_best=True)

        return self.train_eval, self.val_eval, self.test_eval

    def trainSegmentation(self, net, is_eval=True):
        # train
        net.train()

        # optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer = optim.Adam(net.parameters())
        # optimizer = optim.RMSprop(net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay,
        #                           momentum=self.args.momentum)

        # lr_schedule = optim.lr_scheduler.StepLR(optimizer, 20, 0.1)  # 每20个epoch lr=lr*0.1

        loss_f = nn.CrossEntropyLoss() if self.args.num_classes > 1 else nn.BCEWithLogitsLoss()

        for ep in range(1, self.args.epochs + 1):
            net.train()
            iter_loss = 0
            tmp_iter_loss = 0
            for iter, (imgs, targets) in enumerate(self.train_dataloader, start=1):
                assert imgs.shape[1] == self.args.num_channels, "imgs.shape[1]({})".format(
                    imgs.shape[1]) + " != num_channels({})".format(self.args.num_channels)
                imgs = imgs.to(self.args.device, dtype=torch.float32)
                targets = targets.to(self.args.device,
                                     dtype=torch.float32 if self.args.num_classes == 1 else torch.long)

                preds = net(imgs)
                loss = loss_f(preds, targets)
                # + dice_loss(F.softmax(res, dim=1).float(),F.one_hot(targets, self.args.num_classes).permute(0, 3, 1, 2).float(), multiclass=True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_loss += loss.item()
                tmp_iter_loss += loss.item()
                if self.args.verbose and iter % self.args.log_interval == 0:
                    utils.log(log_type="Train",
                              dict_val={"Epoch": ep, "Iter": iter,
                                        "Loss": format(tmp_iter_loss / self.args.log_interval, ".4f")})
                    tmp_iter_loss = .0
            # lr_schedule.step()
            utils.log("Non_Fed", {"Epoch": ep, "Avg_Loss": format(iter_loss / len(self.train_dataloader), ".4f")})

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

    def train(self, global_net, is_eval=True):
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
                w, loss = local.train(copy.deepcopy(global_net).to(self.args.device))
                if self.args.all_clients:
                    local_w[c_id] = copy.deepcopy(w)
                else:
                    local_w.append(copy.deepcopy(w))
                all_local_loss += copy.deepcopy(loss)

            global_loss = all_local_loss / client_num
            global_w = fed.FedAvg(local_w)
            global_net.load_state_dict(global_w)

            utils.log("Fed I.I.D Global", {"Epoch": ep, "Avg_Loss": format(global_loss, ".4f")})

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

    def train(self, local_net):
        # local net
        local_net.train()

        loss_f = nn.CrossEntropyLoss() if self.args.num_classes > 1 else nn.BCEWithLogitsLoss()

        # optimizer = optim.SGD(local_net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer = optim.Adam(local_net.parameters())

        lr_schedule = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)

        ep_loss = .0
        for ep in range(1, self.args.local_ep + 1):
            iter_loss = .0
            tmp_iter_loss = .0
            lr_schedule.step()
            for iter, (imgs, targets) in enumerate(self.train_dataloader, start=1):
                imgs, targets = imgs.to(self.args.device), targets.to(self.args.device)
                optimizer.zero_grad()
                preds = local_net(imgs)
                loss = loss_f(preds, targets)
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
