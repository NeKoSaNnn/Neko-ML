#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import copy
import math

import torch
from torch import nn


class Eval(object):
    def __init__(self, args, dataloader, utils, eval_type="train"):
        self.args = args
        self.utils = utils
        self.dataloader = dataloader
        self.eval_type = eval_type
        self.loss = []
        self.acc = []

        self.best_loss = math.inf
        self.best_acc = .0
        self.best_net = None

        self.name = "{}{}".format("iid-" if args.iid else "", "u" + str(args.num_users) + "-" if args.iid else "") \
                    + "{}-{}-{}-ep{}-{}".format(eval_type, args.dataset, args.model, args.epochs, utils.get_now_time())

    def eval(self, net, get_best=False):
        if self.dataloader.dataset is None:
            pass
        else:
            if self.args.dataset == "mnist" or self.args.dataset == "cifar10":
                return self.evalMultiClassifier(net, get_best)
            elif self.args.dataset == "isic":
                return self.evalISIC(net, get_best)
            else:
                exit("Eval init error!")

    def get_best(self):
        return {"loss": self.best_loss, "acc": self.best_acc, "net": self.best_net}

    def _get_loss(self):
        return self.loss

    def _get_acc(self):
        return self.acc

    def save_best_model(self, only_weight=True):
        assert self.best_net is not None
        self.utils.save_model(self.best_net, save_name=self.name, save_path="./save/pt", only_weight=only_weight)

    def evalMultiClassifier(self, net, get_best=False):  # 10分类
        net.eval()
        eval_loss = .0
        correct = .0
        with torch.no_grad():
            loss_f = nn.CrossEntropyLoss()
            for _, (data, target) in enumerate(self.dataloader):
                data, target = data.to(self.args.device), target.to(self.args.device)
                res = net(data)
                eval_loss += loss_f(res, target).item()
                _, pred_label = torch.max(res.data, 1)
                correct += (pred_label == target).sum().item()
            eval_loss /= len(self.dataloader)
            eval_acc = correct / len(self.dataloader.dataset)
            self.utils.log(self.eval_type, {"Loss": format(eval_loss, ".4f"), "Acc": "{:.2f}%".format(eval_acc * 100)})
            self.loss.append(eval_loss)
            self.acc.append(eval_acc)
            if get_best:
                self.best_net = copy.deepcopy(net) if eval_acc > self.best_acc else self.best_net
                self.best_loss = eval_loss if eval_loss < self.best_loss else self.best_loss
                self.best_acc = eval_acc if eval_acc > self.best_acc else self.best_acc
        return eval_loss, eval_acc

    def evalISIC(self, net, get_best=False):  # 2分类
        net.eval()
        eval_loss = .0
        with torch.no_grad():
            loss_f = nn.BCELoss()
            for _, (data, target) in enumerate(self.dataloader):
                data, target = data.to(self.args.device), target.to(self.args.device)
                res = net(data)
                eval_loss += loss_f(res, target).item()
            eval_loss /= len(self.dataloader)
            self.utils.log(self.eval_type, {"Loss": format(eval_loss, ".4f")})
            self.loss.append(eval_loss)
            if get_best:
                self.best_net = copy.deepcopy(net) if eval_loss < self.best_loss else self.best_net
                self.best_loss = eval_loss if eval_loss < self.best_loss else self.best_loss
        return eval_loss
