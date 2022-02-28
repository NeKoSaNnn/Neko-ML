#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from metrics import dice_coeff, multi_dice_coeff


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

    def eval(self, net, get_best=False):
        if self.dataloader.dataset is None:
            pass
        else:
            if self.args.dataset == "mnist" or self.args.dataset == "cifar10":
                return self.evalMultiClassifier(net, get_best)
            elif self.args.dataset == "isic":
                return self.evalSegmentation(net, get_best)
            else:
                exit("Eval init error!")

    def get_best(self):
        return {"loss": self.best_loss, "acc": self.best_acc, "net": self.best_net}

    def get_loss(self):
        return self.loss

    def get_acc(self):
        return self.acc

    def save_best_model(self, only_weight=True):
        assert self.best_net
        fed_name = "{}{}".format("iid-" if self.args.iid else "",
                                 "u" + str(self.args.num_users) + "-" if self.args.iid else "")
        norm_name = "{}-{}-{}-ep{}-{}".format(self.eval_type, self.args.dataset, self.args.model, self.args.epochs,
                                              self.utils.get_now_time())
        name = fed_name + norm_name
        self.utils.save_model(self.best_net, save_name=name, save_path="./save/pt", only_weight=only_weight)

    def evalMultiClassifier(self, net, get_best=False):  # 10分类
        net.eval()
        eval_loss = 0
        correct = 0
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
            self.utils.log("Eval" + self.eval_type,
                           {"Loss": format(eval_loss, ".4f"), "Acc": "{:.2f}%".format(eval_acc * 100)})
            self.loss.append(eval_loss)
            self.acc.append(eval_acc)
            if get_best:
                self.best_net = copy.deepcopy(net).to(self.args.device) if eval_acc > self.best_acc else self.best_net
                self.best_loss = eval_loss if eval_loss < self.best_loss else self.best_loss
                self.best_acc = eval_acc if eval_acc > self.best_acc else self.best_acc
        net.train()
        return eval_loss, eval_acc

    def evalSegmentation(self, net, get_best=False):  # 2分类
        net.eval()
        dice_score = 0
        eval_loss = 0
        with torch.no_grad():

            loss_f = nn.CrossEntropyLoss() if self.args.num_classes > 1 else nn.BCEWithLogitsLoss()

            for _, (datas, targets) in enumerate(self.dataloader):
                datas = datas.to(self.args.device, dtype=torch.float32)
                targets = targets.to(self.args.device,
                                     dtype=torch.float32 if self.args.num_classes == 1 else torch.long)
                preds = net(datas)

                eval_loss += loss_f(preds, targets).item()

                preds = (preds > 0.5).float()

                if self.args.num_classes == 1:
                    dice_score += dice_coeff(preds, targets).item()
                else:
                    dice_score += F.cross_entropy(preds, targets).item()

                # res = F.softmax(res, dim=1).float()
                # dice_score += dice_loss(res, target, multiclass=True)
                # if self.args.num_classes == 1:
                #     res = (torch.sigmoid(res) > 0.5).float()
                #     dice_score += dice_coeff(res, target).item()
                # else:
                #     target = F.one_hot(target, self.args.num_classes).permute(0, 3, 1, 2).float()
                #     res = F.softmax(res, dim=1)
                # #     res = F.one_hot(res.argmax(dim=1), self.args.num_classes).permute(0, 3, 1, 2).float()
                #     # compute the Dice score, ignoring background
                #     dice_score += multi_dice_coeff(res, target).item()
            dice_score /= len(self.dataloader)
            eval_loss /= len(self.dataloader)
            self.utils.log(self.eval_type, {"Loss": format(eval_loss, ".4f"), "Dice_Score": format(dice_score, ".4f")})
            self.loss.append(eval_loss)
            self.acc.append(dice_score)
            if get_best:
                self.best_net = copy.deepcopy(net).to(self.args.device) if dice_score > self.best_acc else self.best_net
                self.best_loss = eval_loss if eval_loss < self.best_loss else self.best_loss
                self.best_acc = dice_score if dice_score > self.best_acc else self.best_acc
        net.train()
        return eval_loss, dice_score
