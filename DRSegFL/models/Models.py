#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import copy

import numpy as np
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

        self.num_classes = self.config[constants.NUM_CLASSES]
        self.num_channels = self.config[constants.NUM_CHANNELS]

        if constants.TRAIN in self.config:
            self.train_dataset = ListDataset(txt_path=self.config[constants.TRAIN], dataset_name=self.config[constants.NAME_DATASET],
                                             num_classes=self.num_classes, img_size=self.config[constants.IMG_SIZE])
            self.train_dataloader = DataLoader(self.train_dataset,
                                               self.config[constants.BATCH_SIZE], shuffle=True, num_workers=self.config[constants.NUM_WORKERS])
            self.train_contribution = len(self.train_dataset)

        if constants.VALIDATION in self.config:
            self.val_dataset = ListDataset(txt_path=self.config[constants.VALIDATION], dataset_name=self.config[constants.NAME_DATASET],
                                           num_classes=self.num_classes, img_size=self.config[constants.IMG_SIZE])
            self.val_dataloader = DataLoader(self.val_dataset, self.config[constants.EVAL_BATCH_SIZE], shuffle=False, num_workers=1)
            self.val_contribution = len(self.val_dataset)

        if constants.TEST in self.config:
            self.test_dataset = ListDataset(txt_path=self.config[constants.TEST], dataset_name=self.config[constants.NAME_DATASET],
                                            num_classes=self.num_classes, img_size=self.config[constants.IMG_SIZE])
            self.test_dataloader = DataLoader(self.test_dataset, self.config[constants.EVAL_BATCH_SIZE], shuffle=False, num_workers=1)
            self.test_contribution = len(self.test_dataset)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")

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
        self.loss_f = nn.CrossEntropyLoss(ignore_index=self.num_classes) if self.num_classes > 1 else nn.BCEWithLogitsLoss()

    def train(self, epoch=1):
        target_data_type = torch.long if self.num_classes > 1 else torch.float32
        self.net.train()
        ep_losses = []
        for ep in range(1, epoch + 1):
            iter_losses = 0
            for iter, (imgs, targets) in enumerate(self.train_dataloader, start=1):
                imgs = Variable(imgs.to(self.device, torch.float32))
                targets = Variable(targets.to(self.device, target_data_type), requires_grad=False)
                assert imgs.shape[1] == self.num_channels, "imgs.shape[1]({})!=self.num_channels({})".format(imgs.shape[1], self.num_channels)
                # 梯度累计，实现不增大显存而增大batch_size

                preds = self.net(imgs)
                assert preds.shape[1] == self.num_classes, "preds.shape[1]({})!=self.num_classes({})".format(preds.shape[1], self.num_classes)
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
            raise AssertionError("eval_type:{}".format(eval_type))

        target_data_type = torch.long if self.num_classes > 1 else torch.float32
        with torch.no_grad():
            eval_loss = 0
            acc_score = 0
            list_preds = []
            list_targets = []
            for iter, (imgs, targets) in enumerate(eval_dataloader, start=1):
                imgs = Variable(imgs.to(self.device, torch.float32), requires_grad=False)
                targets = Variable(targets.to(self.device, target_data_type), requires_grad=False)

                preds = self.net(imgs)  # [N,C,H,W]
                assert preds.shape[1] == self.num_classes, "{}!={}".format(preds.shape[1], self.num_classes)
                loss = self.loss_f(preds, targets)

                eval_loss += loss.item()

                if self.num_classes == 1:
                    preds = (torch.sigmoid(preds) > 0.5).float()
                    acc_score += metrics.dice_coeff(preds, targets).item()
                else:
                    preds = torch.softmax(preds, dim=1).cpu().numpy()  # [N,C,H,W]
                    preds = np.argmax(preds, axis=1)  # [N,1,H,W]
                    # acc_score += F.cross_entropy(preds, targets, ignore_index=self.num_classes).item()
                    # acc_score += metrics.multi_dice_coeff(preds, targets).item()
                    list_preds.extend(preds[:, ])
                    list_targets.extend(targets[:, ].cpu().numpy())

            eval_loss /= len(eval_dataloader)
            if self.num_classes == 1:
                mDice = acc_score / len(eval_dataloader)
                mIoU = metrics.Dice2IoU(mDice)
                eval_acc = {"mDice": mDice, "mIoU": mIoU}
            else:
                all_acc, accs, ious = metrics.mIoU(list_preds, list_targets, self.num_classes, self.num_classes)
                self.logger.debug("accs={}".format(accs))
                self.logger.debug("ious={}".format(ious))
                mIoU = np.nanmean(ious)
                mAcc = np.nanmean(accs)
                mDice = metrics.IoU2Dice(mIoU)
                eval_acc = {"mIoU": mIoU, "mDice": mDice, "mAcc": mAcc, "all_acc": all_acc}

        self.net.train()
        return eval_loss, eval_acc


class Models:
    unet = unet
