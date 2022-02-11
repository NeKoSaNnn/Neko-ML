#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import os

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from torchvision import transforms

from datasets import ISICDataSet
from models import UNet
from utils.args import args
from utils.utils import utils

utils = utils(log_path="./log")


class predictISIC(object):
    def __init__(self, net, args, save_dir="./save/predict/isic"):
        self.net = net
        self.args = args
        self.save_dir = save_dir
        utils.mkdir_nf(save_dir)

    def predict(self, image_path: str, out_threshold=0.5):
        net = self.net
        net.eval()
        img = ISICDataSet.preprocess(image_path)

        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.to(self.args.device, dtype=torch.float32)

        # unet 多分类通用
        with torch.no_grad():
            output = net(img)
            if self.args.num_classes > 1:
                predict_mask = F.softmax(output, dim=1)[0]
                predict_mask = F.one_hot(predict_mask.argmax(dim=0), self.args.num_classes).permute(2, 0, 1) \
                    .cpu().numpy()

                # （classes,h,w）
            else:
                predict_mask = torch.sigmoid(output)[0]
                predict_mask = (predict_mask > out_threshold).float().cpu().numpy()

        print(predict_mask.shape)
        print(np.sum(predict_mask))
        predict_mask = (predict_mask * 255).astype(np.uint8)
        return predict_mask

    def draw(self, img: Image.Image, target_mask: Image.Image, predict_mask, save=True):
        classes = self.args.num_classes
        fig, ax = plt.subplots(1, classes + 2)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=0)
        ax[0].set_title("ori img")
        ax[0].imshow(img)
        ax[0].axis("off")
        ax[1].set_title("ground truth")
        ax[1].imshow(target_mask)
        ax[1].axis("off")
        # if classes > 1:
        for i in range(classes):
            ax[2 + i].set_title("output mask (class {})".format(i + 1))
            print(predict_mask[i, :, :].shape)
            print(predict_mask[i, :, :].dtype)
            ax[2 + i].imshow(predict_mask[i, :, :], cmap="gray")
            ax[2 + i].axis("off")
        # else:
        #     ax[1].set_title("output mask")
        #     ax[1].imshow(predict_mask)
        #     ax[1].axis("off")
        plt.xticks([]), plt.yticks([])
        if save:
            plt.savefig(os.path.join(self.save_dir, "predict_result_{}.jpg".format(utils.get_now_time())))
            utils.log("Save", {"predict_result_{}.jpg".format(utils.get_now_time()): "success"})
        plt.show()


if __name__ == "__main__":
    # args类初始化
    args = args().get()

    net = UNet.UNet(args.num_channels, args.num_classes).to(args.device)
    net.load_state_dict(torch.load("./save/pt/Test-isic-unet-ep1000-2022-01-17-13-15-26.pt", map_location=args.device))

    test_img_path = "./data/ISIC/val/image/ISIC_0000051.jpg"
    test_mask_path = "./data/ISIC/val/mask/ISIC_0000051_segmentation.png"

    # target = ISICDataSet.target_preprocess(test_mask_path, transforms.ToTensor())
    # plt.figure()
    # plt.imshow(Image.fromarray(target.float().numpy()).convert("L"), cmap="gray")
    # plt.savefig("./save/predict/isic/1.jpg")
    # plt.show()

    predictISIC = predictISIC(net, args)

    predict_mask = predictISIC.predict(test_img_path)

    predictISIC.draw(img=Image.open(test_img_path).resize((256, 256)),
                     target_mask=Image.open(test_mask_path).resize((256, 256)),
                     predict_mask=predict_mask, save=True)
