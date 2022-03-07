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
from PIL import Image
from torch.nn import functional as F

root_dir_name = osp.dirname(sys.path[0])  # ...Neko-ML/
now_dir_name = sys.path[0]  # ...DRSegFL/
sys.path.append(root_dir_name)

from DRSegFL import utils, constants, preprocess
from DRSegFL.models import Models


def predict(config_path, weights_path, predict_img_path, ground_truth_path, out_threshold=0.5):
    assert ".json" in config_path, "config type error :{} ,expect jon".format(config_path)
    now_day = utils.get_now_day()
    now_time = utils.get_now_time()
    config = utils.load_json(config_path)
    img_size = config[constants.IMG_SIZE]
    dataset_name = config[constants.NAME_DATASET]
    model_name = config[constants.NAME_MODEL]
    num_classes = config[constants.NUM_CLASSES]
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predict_dir = osp.join(config[constants.DIR_PREDICT], now_day)
    os.makedirs(predict_dir, exist_ok=True)
    save_path = osp.join(predict_dir, "predict_{}_{}.jpg".format(osp.basename(predict_img_path), now_time))

    if dataset_name == constants.ISIC:
        tensor_img, _, pil_img, pil_gt = preprocess.ISIC_preprocess(predict_img_path, ground_truth_path, img_size)
    elif dataset_name == constants.DDR:
        tensor_img, _, pil_img, pil_gt = preprocess.DDR_preprocess(predict_img_path, ground_truth_path, img_size, num_classes)
    else:
        raise AssertionError("no such dataset:{}".format(dataset_name))
    batch_img = tensor_img.unsqueeze(0)  # (1,Channel,H,W)
    batch_img = batch_img.to(device)

    model = getattr(Models, model_name)(config)
    model.set_weights(weights_path)
    net = model.net
    net.eval()
    with torch.no_grad():
        output = net(batch_img)
        if num_classes > 1:
            predict_mask = F.softmax(output, dim=1)[0]
            predict_mask = F.one_hot(predict_mask.argmax(dim=0), num_classes).permute(2, 0, 1).cpu().numpy()

            # (Classes,H,W)
        else:
            predict_mask = torch.sigmoid(output)[0]
            predict_mask = (predict_mask > out_threshold).float().cpu().numpy()
    predict_mask = (predict_mask * 255).astype(np.uint8)

    utils.draw_predict(num_classes, pil_img, pil_gt, predict_mask, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="path of server config")
    parser.add_argument("--weights_path", type=str, required=True, help="path of weights")
    parser.add_argument("--predict_img_path", type=str, required=True, help="the path of the predict img")
    parser.add_argument("--ground_truth_path", type=str, required=True, help="the path of the ground truth")

    args = parser.parse_args()
    predict(args.config_path, args.weights_path, args.predict_img_path, args.ground_truth_path)
