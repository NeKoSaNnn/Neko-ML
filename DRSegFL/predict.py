#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import argparse
import copy
import glob
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

root_dir_name = osp.dirname(sys.path[0])  # ...Neko-ML/
now_dir_name = sys.path[0]  # ...DRSegFL/
sys.path.append(root_dir_name)

from DRSegFL import utils, constants, preprocess, inference
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

        if constants.SLIDE_INFERENCE in config:
            self.is_slide_inference = True
            self.slide_crop_size = config[constants.SLIDE_INFERENCE][constants.SLIDE_CROP_SIZE]
            self.slide_stride = config[constants.SLIDE_INFERENCE][constants.SLIDE_STRIDE]
        else:
            self.is_slide_inference = False

        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.predict_dir = osp.join(config[constants.DIR_PREDICT], now_day)
        self.batch_save_dir = None
        self.save_dir = None

        self.now_weight_name = "_".join(self.weights_path.rsplit("/", 4)[1:])

        self.model = getattr(Models, model_name)(config)
        self.model.set_weights(weights_path)
        self.net = self.model.net
        self.net.eval()

    def batch_reference(self, predict_img_dir, ground_truth_dir, out_threshold=0.5, num=None, img_suffix="jpg", gt_suffix="png"):
        now_time = utils.get_now_time()
        if constants.TRAIN in predict_img_dir:
            self.batch_save_dir = osp.join(self.predict_dir, self.now_weight_name, constants.TRAIN, "batch", now_time)
        elif constants.VALIDATION in predict_img_dir:
            self.batch_save_dir = osp.join(self.predict_dir, self.now_weight_name, constants.VALIDATION, "batch", now_time)
        elif constants.TEST in predict_img_dir:
            self.batch_save_dir = osp.join(self.predict_dir, self.now_weight_name, constants.TEST, "batch", now_time)
        else:
            self.batch_save_dir = osp.join(self.predict_dir, self.now_weight_name, "batch", now_time)

        os.makedirs(self.batch_save_dir, exist_ok=True)

        img_suffix = "jpg" if img_suffix is None else img_suffix
        gt_suffix = "png" if gt_suffix is None else gt_suffix
        img_paths = sorted(glob.glob(osp.join(predict_img_dir, "*.{}".format(img_suffix))))
        gt_paths = sorted(glob.glob(osp.join(ground_truth_dir, "*.{}".format(gt_suffix))))
        assert len(img_paths) == len(gt_paths), "{}!={}".format(len(img_paths), len(gt_paths))

        _len = len(img_paths)
        if num is None or num > len(img_paths):
            for i in tqdm(range(_len), desc="predicting ...", unit="img"):  # zip tqdm不显示进度条
                self.reference(img_paths[i], gt_paths[i], out_threshold, is_batch=True)
        else:
            random_list = np.random.choice(range(_len), num, replace=False)
            for i in tqdm(range(num), desc="predicting ...", unit="img"):  # zip tqdm不显示进度条
                self.reference(img_paths[random_list[i]], gt_paths[random_list[i]], out_threshold, is_batch=True)

    def reference(self, predict_img_path, ground_truth_path, out_threshold=0.5, is_batch=False):
        now_time = utils.get_now_time()
        if is_batch:
            save_path = osp.join(self.batch_save_dir, "{}.jpg".format(osp.basename(predict_img_path).rsplit(".", 1)[0]))
        else:
            if constants.TRAIN in predict_img_path:
                self.save_dir = osp.join(self.predict_dir, self.now_weight_name, constants.TRAIN)
            elif constants.VALIDATION in predict_img_path:
                self.save_dir = osp.join(self.predict_dir, self.now_weight_name, constants.VALIDATION)
            elif constants.TEST in predict_img_path:
                self.save_dir = osp.join(self.predict_dir, self.now_weight_name, constants.TEST)
            else:
                self.save_dir = osp.join(self.predict_dir, self.now_weight_name)
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = osp.join(self.save_dir, "t{}_{}.jpg".format(now_time, osp.basename(predict_img_path).rsplit(".", 1)[0]))

        # Todo: add dataset , modify belows
        if self.dataset_name == constants.ISIC:
            tensor_img, _, pil_img, pil_gt = preprocess.ISIC_preprocess(predict_img_path, ground_truth_path, self.img_size)
        elif self.dataset_name == constants.DDR:
            tensor_img, _, pil_img, pil_gt = preprocess.DDR_preprocess(predict_img_path, ground_truth_path, self.img_size, is_train=False)
        elif self.dataset_name in [constants.DDR_EX, constants.DDR_HE, constants.DDR_MA, constants.DDR_SE]:
            tensor_img, _, pil_img, pil_gt = preprocess.DDR_OneLesion_preprocess(predict_img_path, ground_truth_path, self.img_size,
                                                                                 is_train=False)
        elif self.dataset_name in [constants.FGARD_EX, constants.FGARD_SE, constants.FGARD_HE]:
            tensor_img, _, pil_img, pil_gt = preprocess.FGADR_OneLesion_preprocess(predict_img_path, ground_truth_path, self.img_size,
                                                                                   is_train=False)
        else:
            raise AssertionError("no such dataset:{}".format(self.dataset_name))
        batch_img = tensor_img.unsqueeze(0)  # (1,Channel,H,W)
        batch_img = batch_img.to(self.device)

        with torch.no_grad():
            if self.is_slide_inference:
                output = inference.slide_inference(batch_img, self.net, self.num_classes, self.slide_crop_size, self.slide_stride)
            else:
                output = inference.whole_inference(batch_img, self.net)

            if self.num_classes > 1:
                predict_mask = F.softmax(output, dim=1)[0]
                predict_mask = F.one_hot(predict_mask.argmax(dim=0), self.num_classes).permute(2, 0, 1).cpu().numpy()

                # (Classes,H,W)
            else:
                predict_mask = torch.sigmoid(output)[0]
                predict_mask = (predict_mask > out_threshold).float().cpu().numpy()
        predict_mask = (predict_mask * 255).astype(np.uint8)

        # Todo: add dataset , modify belows
        if self.dataset_name == constants.ISIC:
            utils.draw_predict(self.classes, pil_img, pil_gt, predict_mask, save_path, verbose=not is_batch)
        elif self.dataset_name == constants.DDR:
            utils.draw_predict(self.classes, pil_img, pil_gt, predict_mask, save_path, draw_gt_one_hot=True, ignore_index=0,
                               verbose=not is_batch)  # ignore bg
        elif self.dataset_name in [constants.DDR_EX, constants.DDR_HE, constants.DDR_MA, constants.DDR_SE, constants.FGARD_HE,
                                   constants.FGARD_SE, constants.FGARD_EX]:
            no_bg_classes = copy.deepcopy(self.classes)
            if "bg" in no_bg_classes:
                no_bg_classes.remove("bg")  # ignore background
            utils.draw_predict(no_bg_classes, pil_img, pil_gt, np.expand_dims(predict_mask[1], 0), save_path, verbose=not is_batch)
        else:
            raise AssertionError("no such dataset:{}".format(self.dataset_name))


if __name__ == "__main__":
    logger = Logger()
    config_path = input("input config_path:").strip()
    weights_path = input("input weights_path:").strip()
    predictor = Predictor(config_path, weights_path)
    is_batch = input("batch predict? (y/n)   ")
    if is_batch.lower() == "y":
        predict_img_dir = input("input predict_img_dir:").strip()
        predict_img_suffix = input("input predict_img_suffix(default is jpg): ").strip()
        ground_truth_dir = input("input ground_truth_dir:").strip()
        ground_truth_suffix = input("input ground_truth_suffix(default is png): ").strip()
        predict_img_suffix = None if len(predict_img_suffix) == 0 else predict_img_suffix
        ground_truth_suffix = None if len(ground_truth_suffix) == 0 else ground_truth_suffix
        is_all = input("predict all ? (y/n)   ")
        if is_all.lower() == "y":
            predictor.batch_reference(predict_img_dir, ground_truth_dir, img_suffix=predict_img_suffix, gt_suffix=ground_truth_suffix)
        else:
            num = int(input("predict num:"))
            predictor.batch_reference(predict_img_dir, ground_truth_dir, num=num, img_suffix=predict_img_suffix, gt_suffix=ground_truth_suffix)
        logger.info("predict over.")
        exit(0)
    else:
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
