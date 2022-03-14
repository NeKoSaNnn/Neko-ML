#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
from torch.nn import functional as F

root_dir_name = osp.dirname(sys.path[0])  # ...Neko-ML/
now_dir_name = sys.path[0]  # ...DRSegFL/
sys.path.append(root_dir_name)

from DRSegFL import utils, constants, preprocess
from DRSegFL.models import Models
from DRSegFL.logger import Logger


class Predictor(object):
    def __init__(self, config_path: str, weights_path: str):
        assert ".json" in config_path, "config type error :{} ,expect jon".format(config_path)
        now_day = utils.get_now_day()

        config = utils.load_json(config_path)
        self.img_size = config[constants.IMG_SIZE]
        self.dataset_name = config[constants.NAME_DATASET]
        model_name = config[constants.NAME_MODEL]
        self.num_classes = config[constants.NUM_CLASSES]
        self.classes = config[constants.CLASSES] if constants.CLASSES in config else None
        self.weights_path = weights_path

        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.predict_dir = osp.join(config[constants.DIR_PREDICT], now_day)
        os.makedirs(self.predict_dir, exist_ok=True)

        self.model = getattr(Models, model_name)(config)
        self.model.set_weights(weights_path)
        self.net = self.model.net
        self.net.eval()

    def reference(self, predict_img_path, ground_truth_path, out_threshold=0.5):
        now_time = utils.get_now_time()
        now_weight_name = "_".join(self.weights_path.rsplit("/", 6)[1:])
        if constants.TRAIN in predict_img_path:
            save_dir = osp.join(self.predict_dir, now_weight_name, constants.TRAIN)
        elif constants.VALIDATION in predict_img_path:
            save_dir = osp.join(self.predict_dir, now_weight_name, constants.VALIDATION)
        elif constants.TEST in predict_img_path:
            save_dir = osp.join(self.predict_dir, now_weight_name, constants.TEST)
        else:
            save_dir = osp.join(self.predict_dir, now_weight_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = osp.join(save_dir, "t{}_{}.jpg".format(now_time, predict_img_path.rsplit(".", 1)[0]))

        if self.dataset_name == constants.ISIC:
            tensor_img, _, pil_img, pil_gt = preprocess.ISIC_preprocess(predict_img_path, ground_truth_path, self.img_size)
        elif self.dataset_name == constants.DDR:
            tensor_img, _, pil_img, pil_gt = preprocess.DDR_preprocess(predict_img_path, ground_truth_path, self.img_size, is_train=False)
        else:
            raise AssertionError("no such dataset:{}".format(self.dataset_name))
        batch_img = tensor_img.unsqueeze(0)  # (1,Channel,H,W)
        batch_img = batch_img.to(self.device)

        with torch.no_grad():
            output = self.net(batch_img)
            if self.num_classes > 1:
                predict_mask = F.softmax(output, dim=1)[0]
                predict_mask = F.one_hot(predict_mask.argmax(dim=0), self.num_classes).permute(2, 0, 1).cpu().numpy()

                # (Classes,H,W)
            else:
                predict_mask = torch.sigmoid(output)[0]
                predict_mask = (predict_mask > out_threshold).float().cpu().numpy()
        predict_mask = (predict_mask * 255).astype(np.uint8)

        if self.dataset_name == constants.ISIC:
            utils.draw_predict(self.classes, pil_img, pil_gt, predict_mask, save_path)
        elif self.dataset_name == constants.DDR:
            utils.draw_predict(self.classes, pil_img, pil_gt, predict_mask, save_path, draw_gt_one_hot=True, ignore_index=0)  # ignore bg
        else:
            raise AssertionError("no such dataset:{}".format(self.dataset_name))


if __name__ == "__main__":
    logger = Logger()
    config_path = input("input config_path:").strip()
    weights_path = input("input weights_path:").strip()
    predictor = Predictor(config_path, weights_path)
    predict_img_path = input("input predict_img_path:").strip()
    ground_truth_path = input("input ground_truth_path:").strip()
    predictor.reference(predict_img_path, ground_truth_path)
    while True:
        is_continue = input("continue? (y/n)   ")
        if is_continue.lower() == "y":
            predict_img_path = input("input predict_img_path:").strip()
            ground_truth_path = input("input ground_truth_path:").strip()
            predictor.reference(predict_img_path, ground_truth_path)
        else:
            logger.info("predict over.")
            exit(0)
