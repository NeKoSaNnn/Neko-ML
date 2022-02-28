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
# sys.path.append(root_dir_name)

from DRSegFL import utils, constants
from DRSegFL.models import Models


def predict(config_path, weights_path, predict_img_path, ground_truth_path, out_threshold=0.5):
    assert ".json" in config_path, "config type error :{} ,expect jon".format(config_path)
    now_day = utils.get_now_day()
    now_time = utils.get_now_time()
    config = utils.load_json(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predict_dir = osp.join(config[constants.DIR_PREDICT], now_day)
    os.makedirs(predict_dir, exist_ok=True)
    save_path = osp.join(predict_dir, "predict_{}_{}.jpg".format(osp.basename(predict_img_path), now_time))

    img = Image.open(predict_img_path)
    img = img.resize((config[constants.IMG_SIZE], config[constants.IMG_SIZE]))
    ground_truth = Image.open(ground_truth_path)
    ground_truth = ground_truth.resize((config[constants.IMG_SIZE], config[constants.IMG_SIZE]))

    _, tensor_img = utils.to_tensor_use_pil(predict_img_path, config[constants.IMG_SIZE])
    tensor_img = tensor_img.unsqueeze(0)
    tensor_img = tensor_img.to(device)

    model = getattr(Models, config[constants.NAME_MODEL])(config)
    model.set_weights(weights_path)
    net = model.net
    net.eval()
    with torch.no_grad():
        output = net(tensor_img)
        if config[constants.NUM_CLASSES] > 1:
            predict_mask = F.softmax(output, dim=1)[0]
            predict_mask = F.one_hot(predict_mask.argmax(dim=0), config[constants.NUM_CLASSES]).permute(2, 0, 1) \
                .cpu().numpy()

            # （classes,h,w）
        else:
            predict_mask = torch.sigmoid(output)[0]
            predict_mask = (predict_mask > out_threshold).float().cpu().numpy()
    predict_mask = (predict_mask * 255).astype(np.uint8)

    utils.draw_predict(config, img, ground_truth, predict_mask, save_path)
    print("predict result saved to {}".format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="path of server config")
    parser.add_argument("--weights_path", type=str, required=True, help="path of weights")
    parser.add_argument("--predict_img_path", type=str, required=True, help="the path of the predict img")
    parser.add_argument("--ground_truth_path", type=str, required=True, help="the path of the ground truth")

    args = parser.parse_args()
    predict(args.config_path, args.weights_path, args.predict_img_path, args.ground_truth_path)
