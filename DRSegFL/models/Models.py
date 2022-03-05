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
from torch.backends.cudnn import benchmark
from torch.utils.data import DataLoader

from DRSegFL import metrics, constants
from DRSegFL.datasets import ListDataset
from DRSegFL.models.UNet import UNet

torch.set_num_threads(4)


class BaseModel(object):
    def __init__(self, config: dict, logger=None):
        self.config = config
        self.logger = logger

        self.logger.info(self.config) if self.logger else print(self.config)

        if constants.TRAIN in self.config:
            self.train_dataset = ListDataset(txt_path=self.config[constants.TRAIN], dataset_name=self.config[constants.NAME_DATASET],
                                             img_size=self.config[constants.IMG_SIZE])
            self.train_dataloader = DataLoader(self.train_dataset,
                                               self.config[constants.BATCH_SIZE], shuffle=True, num_workers=self.config[constants.NUM_WORKERS])
            self.train_contribution = len(self.train_dataset)

        if constants.VALIDATION in self.config:
            self.val_dataset = ListDataset(txt_path=self.config[constants.VALIDATION], dataset_name=self.config[constants.NAME_DATASET],
                                           img_size=self.config[constants.IMG_SIZE])
            self.val_dataloader = DataLoader(self.val_dataset, self.config[constants.EVAL_BATCH_SIZE], shuffle=False, num_workers=1)
            self.val_contribution = len(self.val_dataset)

        if constants.TEST in self.config:
            self.test_dataset = ListDataset(txt_path=self.config[constants.TEST], dataset_name=self.config[constants.NAME_DATASET],
                                            img_size=self.config[constants.IMG_SIZE])
            self.test_dataloader = DataLoader(self.test_dataset, self.config[constants.EVAL_BATCH_SIZE], shuffle=False, num_workers=1)
            self.test_contribution = len(self.test_dataset)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")

        self.num_classes = self.config[constants.NUM_CLASSES]
        self.num_channels = self.config[constants.NUM_CHANNELS]
        self.net = None
        self.optimizer = None
        self.loss_f = None

        self.model_init()

        if self.logger:
            self.logger.info("Model:{} Construct and Init Completed.".format(self.config["model_name"]))
        else:
            print("Model:{} Construct and Init Completed.".format(self.config["model_name"]))

    def __del__(self):
        self.optimizer = None
        del self.optimizer
        self.net = None
        del self.net
        self.loss_f = None
        del self.loss_f
        torch.cuda.empty_cache()

    def get_weights(self):
        return copy.deepcopy(self.net.state_dict())

    def get_opt_weights(self):
        return copy.deepcopy(self.optimizer.state_dict())

    def set_weights(self, weights):
        if isinstance(weights, str) and ".pt" in weights:
            # .pt file
            weights = torch.load(weights)
            self.net.load_state_dict(weights)
        else:
            # state_dict file
            self.net.load_state_dict(copy.deepcopy(weights))

    def model_init(self):
        """
        init belows:
        self.net
        self.optimizer
        self.loss_f
        """
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def eval(self, eval_type):
        raise NotImplementedError


class unet(BaseModel):
    def __init__(self, config: dict, logger=None):
        super(unet, self).__init__(config, logger)

    def model_init(self):
        self.net = UNet(self.num_channels, self.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.loss_f = nn.CrossEntropyLoss() if self.num_classes > 1 else nn.BCEWithLogitsLoss()

    def train(self, epoch=1):
        target_data_type = torch.long if self.num_classes > 1 else torch.float32
        self.net.train()
        ep_losses = []
        for ep in range(1, epoch + 1):
            iter_losses = 0
            for iter, (imgs, targets, _, _) in enumerate(self.train_dataloader, start=1):
                imgs = Variable(imgs.to(self.device, torch.float32))
                targets = Variable(targets.to(self.device, target_data_type), requires_grad=False)
                assert imgs[1] == self.num_channels
                assert targets[1] == self.num_classes
                # 梯度累计，实现不增大显存而增大batch_size

                preds = self.net(imgs)
                loss = self.loss_f(preds, targets)

                iter_losses += loss.item()

                loss.backward()

                if constants.GRAD_ACCUMULATE in self.config and self.config[constants.GRAD_ACCUMULATE] > 0 and iter % \
                        self.config[constants.GRAD_ACCUMULATE] == 0 or constants.GRAD_ACCUMULATE not in self.config or \
                        self.config[constants.GRAD_ACCUMULATE] <= 0:
                    if constants.GRAD_ACCUMULATE in self.config and self.config[constants.GRAD_ACCUMULATE] > 0:
                        if self.logger:
                            self.logger.info(
                                "Accumulate Grad : batch_size*{}".format(self.config[constants.GRAD_ACCUMULATE]))
                        else:
                            print("Accumulate Grad : batch_size*{}".format(self.config[constants.GRAD_ACCUMULATE]))
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if iter % self.config["log_interval_iter"] == 0 or iter == len(self.train_dataloader):
                    if self.logger:
                        self.logger.info(
                            "Train -- LocalEpoch:{} -- iter:{} -- loss:{:.4f}".format(ep, iter, loss.item()))
                    else:
                        print("Train -- LocalEpoch:{} -- iter:{} -- loss:{:.4f}".format(ep, iter, loss.item()))
            ep_losses.append(iter_losses / len(self.train_dataloader))
        return ep_losses

    def eval(self, eval_type):
        eval_loss = 0
        dice_score = 0
        target_data_type = torch.long if self.num_classes > 1 else torch.float32
        self.net.eval()
        if eval_type == constants.TRAIN:
            eval_dataloader = DataLoader(self.train_dataset, self.config[constants.EVAL_BATCH_SIZE], shuffle=False, num_workers=1)
        elif eval_type == constants.VALIDATION:
            eval_dataloader = self.val_dataloader
        elif eval_type == constants.TEST:
            eval_dataloader = self.test_dataloader
        else:
            if self.logger:
                self.logger.error("Error Eval_type:{}".format(eval_type))
            else:
                print("Error Eval_type:{}".format(eval_type))
            return eval_loss, dice_score

        with torch.no_grad():
            for iter, (imgs, targets, _, _) in enumerate(eval_dataloader, start=1):
                imgs = Variable(imgs.to(self.device, torch.float32), requires_grad=False)
                targets = Variable(targets.to(self.device, target_data_type), requires_grad=False)

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
