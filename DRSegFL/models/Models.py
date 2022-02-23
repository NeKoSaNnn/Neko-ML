#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from DRSegFL import metrics
from DRSegFL.datasets import ListDataset
from UNet import UNet

torch.set_num_threads(4)


class unet(object):
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.logger.info(self.config)
        self.train_dataset = ListDataset(txt_path=self.config["train"],
                                         img_type=".jpg",
                                         target_type=".jpg",
                                         img_size=256, is_augment=True)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           self.config["batch_size"],
                                           shuffle=True,
                                           num_workers=self.config["num_workers"])
        if "val" in self.config:
            self.val_dataset = ListDataset(txt_path=self.config["val"],
                                           img_type=".jpg",
                                           target_type=".jpg",
                                           img_size=256)
            self.val_dataloader = DataLoader(self.val_dataset,
                                             self.config["eval_batch_size"],
                                             shuffle=False,
                                             num_workers=1)
        self.test_dataset = ListDataset(txt_path=self.config["test"],
                                        img_type=".jpg",
                                        target_type=".jpg",
                                        img_size=256)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          self.config["eval_batch_size"],
                                          shuffle=False,
                                          num_workers=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = UNet(self.config["num_channels"], self.config["num_classes"]).to(self.device)
        self.logger.info("model:{} construct complete".format(self.config["model_name"]))
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.loss_f = nn.CrossEntropyLoss() if self.config["num_classes"] > 1 else nn.BCEWithLogitsLoss()

    def get_weights(self):
        return [param.data.cpu().numpy() for param in self.net.parameters()]

    def set_weights(self, params):
        for i, param in enumerate(self.net.parameters()):
            new_param = torch.from_numpy(params[i]).cuda() if torch.cuda.is_available() else torch.from_numpy(params[i])
            param.data.copy_(new_param)

    def train(self, epoch=1):
        self.net.train()
        ep_losses = []
        for ep in range(1, epoch + 1):
            iter_losses = 0
            for iter, (imgs, targets, _) in enumerate(self.train_dataloader, start=1):
                imgs, targets = Variable(imgs.to(self.device)), Variable(targets.to(self.device), requires_grad=False)
                # 梯度累计，实现不增大显存而增大batch_size

                preds = self.net(imgs)
                loss = self.loss_f(preds, targets)

                iter_losses += loss.item()

                loss.backward()

                if "accumulate_grad" in self.config and self.config["accumulate_grad"] > 0 and iter % self.config[
                    "accumulate_grad"] == 0 \
                        or "accumulate_grad" not in self.config \
                        or self.config["accumulate_grad"] <= 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if iter % self.config["log_interval"] == 0 or iter == len(self.train_dataloader):
                    self.logger.info("LocalEpoch:{} -- iter:{} -- loss:{:.4f}".format(ep, iter, loss.item()))
            ep_losses.append(iter_losses / len(self.train_dataloader))
        return ep_losses

    def eval(self):
        eval_loss = 0
        dice_score = 0
        self.net.eval()
        for iter, (imgs, targets, _) in enumerate(self.val_dataloader, start=1):
            imgs, targets = Variable(imgs.to(self.device), requires_grad=False), \
                            Variable(targets.to(self.device), requires_grad=False)
            # 梯度累计，实现不增大显存而增大batch_size

            with torch.no_grad():
                preds = self.net(imgs)
                loss = self.loss_f(preds, targets)

                eval_loss += loss.item()
                preds = (preds > 0.5).float()

                if self.config["num_classes"] == 1:
                    dice_score += metrics.dice_coeff(preds, targets).item()
                else:
                    dice_score += F.cross_entropy(preds, targets).item()
        eval_loss /= len(self.val_dataloader)
        dice_score /= len(self.val_dataloader)
        self.net.train()
        return eval_loss, dice_score

    def test(self):
        test_loss = 0
        dice_score = 0
        self.net.eval()
        for iter, (imgs, targets, _) in enumerate(self.test_dataloader, start=1):
            imgs, targets = Variable(imgs.to(self.device), requires_grad=False), \
                            Variable(targets.to(self.device), requires_grad=False)
            # 梯度累计，实现不增大显存而增大batch_size

            with torch.no_grad():
                preds = self.net(imgs)
                loss = self.loss_f(preds, targets)

                test_loss += loss.item()
                preds = (preds > 0.5).float()

                if self.config["num_classes"] == 1:
                    dice_score += metrics.dice_coeff(preds, targets).item()
                else:
                    dice_score += F.cross_entropy(preds, targets).item()
        test_loss /= len(self.test_dataloader)
        dice_score /= len(self.test_dataloader)
        self.net.train()
        return test_loss, dice_score


class Models:
    unet = unet
