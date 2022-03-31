#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import copy
import sys

import numpy as np
import torch
import torch.nn as nn
import os.path as osp
import torch.nn.functional as F
from torch.autograd import Variable
from torch.backends.cudnn import benchmark
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
from opacus import PrivacyEngine
from DRSegFL import metrics, constants, inference
from DRSegFL.datasets import ListDataset
from DRSegFL.logger import Logger
from DRSegFL.loss import *
from DRSegFL.models.UNet import UNet

torch.set_num_threads(4)


class BaseModel(object):
    def __init__(self, config: dict, logger=None, only_init_weights=False, last_epoch=-1):
        self.config = config
        self.logger = Logger(logger)

        self.num_classes = self.config[constants.NUM_CLASSES]
        self.num_channels = self.config[constants.NUM_CHANNELS]
        self.dataset_name = self.config[constants.NAME_DATASET]
        self.classes = self.config[constants.CLASSES]
        self.DP = self.config[constants.DP] if constants.DP in self.config else False
        if constants.SLIDE_INFERENCE in self.config:
            self.is_slide_inference = True
            self.slide_crop_size = self.config[constants.SLIDE_INFERENCE][constants.SLIDE_CROP_SIZE]
            self.slide_stride = self.config[constants.SLIDE_INFERENCE][constants.SLIDE_STRIDE]
        else:
            self.is_slide_inference = False

        if constants.TRAIN in self.config:
            if self.is_slide_inference:
                self.train_dataset = ListDataset(txt_path=self.config[constants.TRAIN], dataset_name=self.config[constants.NAME_DATASET],
                                                 num_classes=self.num_classes, img_size=self.slide_crop_size, is_train=True)
            else:
                self.train_dataset = ListDataset(txt_path=self.config[constants.TRAIN], dataset_name=self.config[constants.NAME_DATASET],
                                                 num_classes=self.num_classes, img_size=self.config[constants.IMG_SIZE], is_train=False)
            self.train_dataloader = DataLoader(self.train_dataset, self.config[constants.BATCH_SIZE], shuffle=True,
                                               num_workers=self.config[constants.NUM_WORKERS], drop_last=True)  # drop last for BN
            self.train_contribution = len(self.train_dataloader)

        if constants.VALIDATION in self.config:
            self.val_dataset = ListDataset(txt_path=self.config[constants.VALIDATION], dataset_name=self.config[constants.NAME_DATASET],
                                           num_classes=self.num_classes, img_size=self.config[constants.IMG_SIZE], is_train=False)
            self.val_dataloader = DataLoader(self.val_dataset, self.config[constants.EVAL_BATCH_SIZE], shuffle=False,
                                             num_workers=self.config[constants.NUM_WORKERS])
            self.val_contribution = len(self.val_dataloader)

        if constants.TEST in self.config:
            self.test_dataset = ListDataset(txt_path=self.config[constants.TEST], dataset_name=self.config[constants.NAME_DATASET],
                                            num_classes=self.num_classes, img_size=self.config[constants.IMG_SIZE], is_train=False)
            self.test_dataloader = DataLoader(self.test_dataset, self.config[constants.EVAL_BATCH_SIZE], shuffle=False,
                                              num_workers=self.config[constants.NUM_WORKERS])
            self.test_contribution = len(self.test_dataloader)

        self.net = None
        self.optimizer = None
        self.loss_f = []
        self.loss_weight = None
        self.schedule = None

        self.logger.info("Model:{} init ......".format(self.config[constants.NAME_MODEL]))

        if only_init_weights:
            self.net_init(device=torch.device("cpu"))
            # self.net = PrivacyEngine.get_compatible_module(self.net) if self.DP else self.net
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                # torch.backends.cudnn.benchmark = True
            else:
                self.device = torch.device("cpu")

            self.net_init(device=self.device)
            self.loss_init()

            self.logger.info("Model:{} Construct and Init Completed.".format(self.config[constants.NAME_MODEL]))

            # self.net = PrivacyEngine.get_compatible_module(self.net) if self.DP else self.net

            if "optim" in self.config.keys() and self.config["optim"] is not None:
                if self.config["optim"]["type"] == "SGD":
                    self.optimizer = torch.optim.SGD([{"params": self.net.parameters(), "initial_lr": float(self.config["optim"]["lr"])}],
                                                     lr=float(self.config["optim"]["lr"]),
                                                     momentum=float(self.config["optim"]["momentum"]),
                                                     weight_decay=float(self.config["optim"]["weight_decay"]))
            else:
                # default is adam optimizer
                self.optimizer = torch.optim.Adam([{"params": self.net.parameters(), "initial_lr": float(self.config["optim"]["lr"])}],
                                                  lr=float(self.config["optim"]["lr"]))

            if "lr_schedule" in self.config.keys() and self.config["lr_schedule"] is not None:
                self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                           T_max=int(self.config["lr_schedule"]["t_max"]),
                                                                           eta_min=float(self.config["lr_schedule"]["min_lr"]),
                                                                           last_epoch=last_epoch)
                # self.schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                #                                                            mode=self.config["lr_schedule"]["mode"],
                #                                                            factor=float(self.config["lr_schedule"]["factor"]),
                #                                                            patience=int(self.config["lr_schedule"]["patience"]),
                #                                                            min_lr=float(self.config["lr_schedule"]["min_lr"]))

            # Difference Privacy
            # if self.DP:
            #     max_per_sample_grad_norm = 1.0
            #     privacy_engine = PrivacyEngine(accountant="gdp")
            #     n_layers = len([(n, p) for n, p in self.net.named_parameters() if p.requires_grad])
            #     max_grad_norm = [max_per_sample_grad_norm / np.sqrt(n_layers)] * n_layers
            #     self.net, self.optimizer, self.train_dataloader = privacy_engine.make_private(
            #         module=self.net,
            #         optimizer=self.optimizer,
            #         data_loader=self.train_dataloader,
            #         noise_multiplier=1.1,
            #         max_grad_norm=max_grad_norm,
            #         clipping="per_layer")

    def __del__(self):
        self.optimizer = None
        del self.optimizer
        self.net = None
        del self.net
        self.loss_f = None
        del self.loss_f
        torch.cuda.empty_cache()

    def get_weights(self, to_cpu=False, DP=False):
        if DP:
            for name, param in self.net.named_parameters():
                if param.requires_grad:
                    noise = torch.normal(0, 1, param.shape, requires_grad=True).to(param.device)
                    param = param + noise
                    self.logger.debug("{}({},{}):add noise".format(name, param.shape, param.dtype))
        return copy.deepcopy(self.net).cpu().state_dict() if to_cpu else copy.deepcopy(self.net).state_dict()

    def get_opt_weights(self):
        return copy.deepcopy(self.optimizer).state_dict()

    def set_weights(self, weights, is_cpu=False):
        if isinstance(weights, str) and (".pt" in weights or ".pth" in weights):
            # .pt/.pth file
            weights = torch.load(weights, map_location=self.device)
            self.net.load_state_dict(weights, strict=False)
        else:
            # state_dict file
            if is_cpu:
                self.net = self.net.to("cpu")
                self.net.load_state_dict(copy.deepcopy(weights), strict=False)
                self.net = self.net.to(self.device)
            else:
                self.net.load_state_dict(copy.deepcopy(weights), strict=False)

    def loss_init(self):
        self.loss_f = []
        # Todo: add dataset , modify belows
        if self.dataset_name == constants.ISIC:
            self.loss_weight = 1
            self.loss_f.append(nn.CrossEntropyLoss())
        elif self.dataset_name == constants.DDR:
            self.loss_weight = 1
            self.loss_f.append(FocalLoss(np.array([1.0, 3.0, 3.0, 3.0, 3.0])))
            # class_weights = torch.FloatTensor([0.01, 1., 1., 1., 1.]).to(self.device)
            # self.loss_f.append(nn.CrossEntropyLoss(weight=class_weights))
            # self.loss_f.append(smp.losses.DiceLoss(mode=smp.losses.constants.MULTICLASS_MODE, ignore_index=0))
            # self.loss_f.append(smp.losses.SoftBCEWithLogitsLoss(pos_weight=class_weights))
            # self.loss_f.append(CrossEntropyLoss(class_weight=[0.01, 1., 1., 1., 1.]))
            # self.loss_f.append(BinaryLoss(loss_type="dice", class_weight=[0.01, 1., 1., 1., 1.]))
            # self.loss_f.append(FocalLoss(gamma=2.0, alpha=0.25, class_weight=[0.01, 1., 1., 1., 1.]))
            # self.loss_f.append(loss.FocalLoss(gamma=2, alpha=0.25, class_weight=class_weights))
            # self.loss_f.append(smp.losses.FocalLoss(mode=smp.losses.constants.MULTICLASS_MODE, alpha=0.25, gamma=2))
            # self.loss_f.append(smp.losses.DiceLoss(mode=smp.losses.constants.MULTICLASS_MODE, ignore_index=0))
            # self.loss_f.append(smp.losses.SoftCrossEntropyLoss(smooth_factor=0, ignore_index=0))
            # self.loss_f.append(smp.losses.SoftBCEWithLogitsLoss(pos_weight=class_weights))
        elif self.dataset_name in [constants.DDR_EX, constants.DDR_HE, constants.DDR_SE]:
            self.loss_weight = 1
            self.loss_f.append(FocalLoss(alpha=np.array([1.0, 3.0])))
            # self.loss_f.append(smp.losses.SoftBCEWithLogitsLoss(pos_weight=torch.FloatTensor([100.]).to(self.device)))
            # self.loss_f.append(nn.CrossEntropyLoss(weight=torch.FloatTensor([0.01, 1.]).to(self.device)))
            # self.loss_f.append(smp.losses.DiceLoss(mode=smp.losses.constants.MULTICLASS_MODE))
            # self.loss_f.append(smp.losses.FocalLoss(mode=smp.losses.constants.BINARY_MODE, alpha=3, gamma=2))
        elif self.dataset_name == constants.DDR_MA:
            self.loss_weight = 1
            class_weights = torch.FloatTensor([0.01, 1.]).to(self.device)
            self.loss_f.append(nn.CrossEntropyLoss(weight=class_weights))
        else:
            raise AssertionError("dataset error:{}".format(self.dataset_name))
        assert self.loss_weight is None \
               or isinstance(self.loss_weight, (int, float)) \
               or isinstance(self.loss_weight, list) and (len(self.loss_f) == len(self.loss_weight))

    def cal_loss(self, pred, target, weight=None):
        """
        :param pred:
        :param target:
        :param weight: list , int or float , if is None , weight=1
        :return:
        """
        if len(self.loss_f) == 0:
            self.loss_init()
            return self.cal_loss(pred, target, weight)

        if weight is not None:
            assert isinstance(weight, list) and len(weight) == len(self.loss_f) or isinstance(weight, (int, float))
            if isinstance(weight, (int, float)):
                weight = [weight] * len(self.loss_f)
            weight = torch.as_tensor(weight)
            loss = self.loss_f[0](pred, target) * weight[0]
            for i, loss_func in enumerate(self.loss_f[1:], start=1):
                loss += loss_func(pred, target) * weight[i]
        else:
            loss = self.loss_f[0](pred, target)
            for i, loss_func in enumerate(self.loss_f[1:], start=1):
                loss += loss_func(pred, target)
        return loss

    def net_init(self, device):
        """
        init self.net:
        """
        raise NotImplementedError

    def train(self, epoch=1, tbX=None, sio=None):  # target_data_type = torch.long if self.num_classes > 1 else torch.float32
        self.net = self.net.to(self.device)
        self.net.train()
        ep_losses = []
        for ep in range(1, epoch + 1):
            self.logger.info("Train -- LocalEpoch:{} -- lr:{}".format(ep, self.optimizer.state_dict()['param_groups'][0]['lr']))
            if self.schedule:
                self.logger.info("Train -- LocalEpoch:{} -- last_epoch:{}".format(ep, self.schedule.last_epoch))
            iter_losses = 0
            for iter, (imgs, targets) in enumerate(self.train_dataloader, start=1):
                imgs = Variable(imgs.to(self.device, torch.float32))
                targets = Variable(targets.to(self.device, torch.long), requires_grad=False)
                assert imgs.shape[1] == self.num_channels, "imgs.shape[1]({})!=self.num_channels({})".format(imgs.shape[1], self.num_channels)

                preds = self.net(imgs)
                assert preds.shape[1] == self.num_classes, "preds.shape[1]({})!=self.num_classes({})".format(preds.shape[1], self.num_classes)
                loss = self.cal_loss(preds, targets, weight=self.loss_weight)

                iter_losses += loss.item()

                loss.backward()

                # grad_accumulate，increase batch_size without increasing gpu
                if constants.GRAD_ACCUMULATE in self.config and self.config[constants.GRAD_ACCUMULATE] > 0 and iter % \
                        self.config[constants.GRAD_ACCUMULATE] == 0 or constants.GRAD_ACCUMULATE not in self.config or \
                        self.config[constants.GRAD_ACCUMULATE] <= 0:
                    if constants.GRAD_ACCUMULATE in self.config and self.config[constants.GRAD_ACCUMULATE] > 0:
                        self.logger.info("Accumulate Grad : batch_size*{}".format(self.config[constants.GRAD_ACCUMULATE]))
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if iter % self.config["log_interval_iter"] == 0 or iter == len(self.train_dataloader):
                    self.logger.info("Train -- LocalEpoch:{} -- iter:{} -- loss:{:.4f}".format(ep, iter, loss.item()))
                if sio is not None:
                    sio.emit("train_process", {"process": "{:.1f}%".format((ep + iter / len(self.train_dataloader) - 1) / epoch * 100)})
            ep_loss = iter_losses / len(self.train_dataloader)
            if tbX is not None:
                tbX.add_scalar("train/local_loss", ep_loss, self.schedule.last_epoch + 1)
            ep_losses.append(ep_loss)
            if self.schedule is not None:
                self.schedule.step()
        return ep_losses

    def eval(self, eval_type, is_global_eval, sio=None):
        self.net = self.net.to(self.device)
        self.net.eval()
        if eval_type == constants.TRAIN:
            train_dataset = ListDataset(txt_path=self.config[constants.TRAIN], dataset_name=self.config[constants.NAME_DATASET],
                                        num_classes=self.num_classes, img_size=self.config[constants.IMG_SIZE], is_train=False)
            eval_dataloader = DataLoader(train_dataset, self.config[constants.EVAL_BATCH_SIZE], shuffle=False,
                                         num_workers=self.config[constants.NUM_WORKERS])
        elif eval_type == constants.VALIDATION:
            eval_dataloader = self.val_dataloader
        elif eval_type == constants.TEST:
            eval_dataloader = self.test_dataloader
        else:
            raise AssertionError("eval_type:{}".format(eval_type))

        # target_data_type = torch.long if self.num_classes > 1 else torch.float32
        with torch.no_grad():
            eval_loss = 0
            # acc_score = 0
            list_preds = []
            list_targets = []
            for iter, (imgs, targets) in enumerate(eval_dataloader, start=1):
                imgs = Variable(imgs.to(self.device, torch.float32), requires_grad=False)
                targets = Variable(targets.to(self.device, torch.long), requires_grad=False)

                # preds = self.net(imgs)  # [N,C,H,W]
                if self.is_slide_inference:
                    preds = inference.slide_inference(imgs, self.net, self.num_classes, self.slide_crop_size, self.slide_stride)
                else:
                    preds = inference.whole_inference(imgs, self.net)

                assert preds.shape[1] == self.num_classes, "{}!={}".format(preds.shape[1], self.num_classes)
                loss = self.cal_loss(preds, targets, weight=self.loss_weight)

                eval_loss += loss.item()

                if self.num_classes == 1:
                    preds = (torch.sigmoid(preds) > 0.5).long().cpu().numpy()
                    # acc_score += metrics.dice_coeff(preds, targets).item()
                else:
                    preds = torch.softmax(preds, dim=1).cpu().numpy()  # [N,C,H,W]
                    preds = np.argmax(preds, axis=1)  # [N,1,H,W]
                    # acc_score += F.cross_entropy(preds, targets, ignore_index=self.num_classes).item()
                    # acc_score += metrics.multi_dice_coeff(preds, targets).item()
                list_preds.extend(preds[:, ])
                list_targets.extend(targets[:, ].cpu().numpy())

                if is_global_eval and sio is not None:
                    sio.emit("eval_process", {"process": "{:.1f}%".format(iter / len(eval_dataloader) * 100), "type": eval_type})

            eval_loss /= len(eval_dataloader)
            if self.num_classes == 1:
                all_acc, accs, ious, dices, f_scores = metrics.cal_metric(list_preds, list_targets, 2, -1, multi=100)
                # mDice = acc_score / len(eval_dataloader)
                # mIoU = metrics.Dice2IoU(mDice)
                # eval_acc = {"mDice": mDice, "mIoU": mIoU}
            else:
                all_acc, accs, ious, dices, f_scores = metrics.cal_metric(list_preds, list_targets, self.num_classes, -1, multi=100)
            self.logger.debug("accs  =    {}".format(np.around(accs, 4)))
            self.logger.debug("ious  =    {}".format(np.around(ious, 4)))
            self.logger.debug("dices =    {}".format(np.around(dices, 4)))
            self.logger.debug("f_scores = {}".format(np.around(f_scores, 4)))
            mIoU = np.around(np.nanmean(ious), 4)
            mAcc = np.around(np.nanmean(accs), 4)
            mDice = np.around(np.nanmean(dices), 4)
            F_score = np.around(np.nanmean(f_scores), 4)
            all_acc = np.around(all_acc, 4)
            class_iou = {}
            class_dice = {}
            for i, c in enumerate(self.classes):
                class_iou["{}_IoU".format(c)] = np.around(np.nan_to_num(ious[i]), 4)
                class_dice["{}_Dice".format(c)] = np.around(np.nan_to_num(dices[i]), 4)
            eval_acc = {"mIoU": mIoU, "mDice": mDice, "F_score": F_score, "mAcc": mAcc, "all_acc": all_acc}
            eval_acc.update(class_iou)
            eval_acc.update(class_dice)
        self.net.train()
        return eval_loss, eval_acc


class unet(BaseModel):
    def __init__(self, config: dict, logger=None, only_init_weights=False, last_epoch=-1):
        super(unet, self).__init__(config, logger, only_init_weights, last_epoch)

    def net_init(self, device):
        self.net = UNet(self.num_channels, self.num_classes).to(device)


class res_unet(BaseModel):
    def __init__(self, config: dict, logger=None, only_init_weights=False, last_epoch=-1):
        super(res_unet, self).__init__(config, logger, only_init_weights, last_epoch)

    def net_init(self, device):
        self.net = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=self.num_channels,
            classes=self.num_classes).to(device)
        # self.net = UNet(self.num_channels, self.num_classes).to(self.device)


class dense_unet(BaseModel):
    def __init__(self, config: dict, logger=None, only_init_weights=False, last_epoch=-1):
        super(dense_unet, self).__init__(config, logger, only_init_weights, last_epoch)

    def net_init(self, device):
        self.net = smp.Unet(
            encoder_name="densenet169",
            encoder_weights="imagenet",
            in_channels=self.num_channels,
            classes=self.num_classes).to(device)


class unetplusplus(BaseModel):
    def __init__(self, config: dict, logger=None, only_init_weights=False, last_epoch=-1):
        super(unetplusplus, self).__init__(config, logger, only_init_weights, last_epoch)

    def net_init(self, device):
        self.net = smp.UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=self.num_channels,
            classes=self.num_classes).to(device)


class deeplabv3(BaseModel):
    def __init__(self, config: dict, logger=None, only_init_weights=False, last_epoch=-1):
        super(deeplabv3, self).__init__(config, logger, only_init_weights, last_epoch)

    def net_init(self, device):
        self.net = smp.DeepLabV3(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=self.num_channels,
            classes=self.num_classes).to(device)


class deeplabv3plus(BaseModel):
    def __init__(self, config: dict, logger=None, only_init_weights=False, last_epoch=-1):
        super(deeplabv3plus, self).__init__(config, logger, only_init_weights, last_epoch)

    def net_init(self, device):
        self.net = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=self.num_channels,
            classes=self.num_classes).to(device)


class fpn(BaseModel):
    def __init__(self, config: dict, logger=None, only_init_weights=False, last_epoch=-1):
        super(fpn, self).__init__(config, logger, only_init_weights, last_epoch)

    def net_init(self, device):
        self.net = smp.FPN(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=self.num_channels,
            classes=self.num_classes).to(device)


class pspnet(BaseModel):
    def __init__(self, config: dict, logger=None, only_init_weights=False, last_epoch=-1):
        super(pspnet, self).__init__(config, logger, only_init_weights, last_epoch)

    def net_init(self, device):
        self.net = smp.PSPNet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=self.num_channels,
            classes=self.num_classes).to(device)


class Models:
    unet = unet
    res_unet = res_unet
    dense_unet = dense_unet
    unetplusplus = unetplusplus
    deeplabv3 = deeplabv3
    deeplabv3plus = deeplabv3plus
    fpn = fpn
    pspnet = pspnet
