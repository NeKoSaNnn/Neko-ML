#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import torch
import torch.nn.functional as F


class Eval(object):
    def __init__(self, args, dataloader, utils):
        self.args = args
        self.utils = utils
        self.dataloader = dataloader

    def eval(self, net, log_type="Test"):
        net.eval()
        eval_loss = .0
        correct = .0
        with torch.no_grad():
            for _, (data, label) in enumerate(self.dataloader):
                data, label = data.to(self.args.device), label.to(self.args.device)
                res = net(data)
                eval_loss += F.cross_entropy(res, label, reduction="sum").item()  # 方法一
                # eval_loss += nn.CrossEntropyLoss(res, label).item()  # 方法二
                _, pred_label = torch.max(res.data, 1)
                correct += (pred_label == label).sum().item()
        eval_loss /= len(self.dataloader.dataset)  # 方法一
        # eval_loss /= len(self.dataloader)  # 方法二
        eval_acc = correct / len(self.dataloader.dataset)
        self.utils.log(log_type, {"Loss": format(eval_loss, ".4f"), "Acc": "{:.2f}%".format(eval_acc * 100)})
        return eval_loss, eval_acc
