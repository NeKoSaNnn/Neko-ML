#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from DRSegFL import metrics, constants
from DRSegFL.datasets import ListDataset
from DRSegFL.models.UNet import UNet

torch.set_num_threads(4)


class unet(object):
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.logger.info(self.config)
        if constants.TRAIN in self.config:
            self.train_dataset = ListDataset(txt_path=self.config[constants.TRAIN],
                                             img_size=self.config[constants.IMG_SIZE], is_augment=True)
            self.train_dataloader = DataLoader(self.train_dataset,
                                               self.config[constants.BATCH_SIZE],
                                               shuffle=True,
                                               num_workers=self.config[constants.NUM_WORKERS])
            self.train_contribution = len(self.train_dataset)

        if constants.VALIDATION in self.config:
            self.val_dataset = ListDataset(txt_path=self.config[constants.VALIDATION],
                                           img_size=self.config[constants.IMG_SIZE])
            self.val_dataloader = DataLoader(self.val_dataset,
                                             self.config[constants.EVAL_BATCH_SIZE],
                                             shuffle=False,
                                             num_workers=1)
            self.val_contribution = len(self.val_dataset)

        if constants.TEST in self.config:
            self.test_dataset = ListDataset(txt_path=self.config[constants.TEST],
                                            img_size=self.config[constants.IMG_SIZE])
            self.test_dataloader = DataLoader(self.test_dataset,
                                              self.config[constants.EVAL_BATCH_SIZE],
                                              shuffle=False,
                                              num_workers=1)
            self.test_contribution = len(self.test_dataset)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = UNet(self.config[constants.NUM_CHANNELS], self.config[constants.NUM_CLASSES]).to(self.device)
        self.logger.info("model:{} construct complete".format(self.config["model_name"]))
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.loss_f = nn.CrossEntropyLoss() if self.config[constants.NUM_CLASSES] > 1 else nn.BCEWithLogitsLoss()

    def get_weights(self):
        return self.net.state_dict()

    def set_weights(self, weights):
        self.net.load_state_dict(copy.deepcopy(weights))

    def train(self, epoch=1):
        self.logger.info("local train start ...")
        self.net.train()
        ep_losses = []
        for ep in range(1, epoch + 1):
            iter_losses = 0
            for iter, (imgs, targets, _, _) in enumerate(self.train_dataloader, start=1):
                imgs, targets = Variable(imgs.to(self.device, torch.float32)), Variable(
                    targets.to(self.device, torch.float32), requires_grad=False)
                # 梯度累计，实现不增大显存而增大batch_size

                preds = self.net(imgs)
                loss = self.loss_f(preds, targets)

                iter_losses += loss.item()

                loss.backward()

                if constants.GRAD_ACCUMULATE in self.config and self.config[constants.GRAD_ACCUMULATE] > 0 and iter % \
                        self.config[constants.GRAD_ACCUMULATE] == 0 or constants.GRAD_ACCUMULATE not in self.config or \
                        self.config[constants.GRAD_ACCUMULATE] <= 0:
                    if constants.GRAD_ACCUMULATE in self.config and self.config[constants.GRAD_ACCUMULATE] > 0:
                        self.logger.info(
                            "accumulate grad : batch_size*{}".format(self.config[constants.GRAD_ACCUMULATE]))
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if iter % self.config["log_interval_iter"] == 0 or iter == len(self.train_dataloader):
                    self.logger.info("Train -- LocalEpoch:{} -- iter:{} -- loss:{:.4f}".format(ep, iter, loss.item()))
            ep_losses.append(iter_losses / len(self.train_dataloader))
        return ep_losses

    def eval(self, eval_type):
        eval_loss = 0
        dice_score = 0
        self.net.eval()
        if eval_type == constants.VALIDATION:
            eval_dataloader = self.val_dataloader
        elif eval_type == constants.TEST:
            eval_dataloader = self.test_dataloader
        else:
            self.logger.error("eval_type:{} error!".format(eval_type))
            return eval_loss, dice_score

        for iter, (imgs, targets, _, _) in enumerate(eval_dataloader, start=1):
            imgs, targets = Variable(imgs.to(self.device, torch.float32), requires_grad=False), \
                            Variable(targets.to(self.device, torch.float32), requires_grad=False)
            with torch.no_grad():
                preds = self.net(imgs)
                loss = self.loss_f(preds, targets)

                eval_loss += loss.item()
                preds = (preds > 0.5).float()

                if self.config[constants.NUM_CLASSES] == 1:
                    dice_score += metrics.dice_coeff(preds, targets).item()
                else:
                    dice_score += F.cross_entropy(preds, targets).item()

        eval_loss /= len(eval_dataloader)
        dice_score /= len(eval_dataloader)
        self.net.train()
        return eval_loss, dice_score


class Models:
    unet = unet
