#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
from torchvision import transforms
from torch import nn

from datasets import InitDataSet
from models import MLP, CNN, UNet
from train import Train, GlobalTrain
from utils.args import args
from utils.utils import utils

if __name__ == "__main__":
    # args类初始化
    args = args().get()

    # utils工具类初始化
    utils = utils(log_path="./log")
    utils.log("Config", {"args": args})

    name = "{}{}".format("iid-" if args.iid else "", "u" + str(args.num_users) + "-" if args.iid else "") \
           + "{}-{}-ep{}-{}".format(args.dataset, args.model, args.epochs, utils.get_now_time())

    # 初始化网络
    net = None
    if args.model == "mlp":
        net = MLP.MLP(args.input_size, 200, args.num_classes).to(args.device)
    elif args.model == "cnn":
        if args.dataset == "mnist":
            net = CNN.MnistCNN(args).to(args.device)
        elif args.dataset == "cifar10":
            net = CNN.CifarCNN(args).to(args.device)
    elif args.model == "unet":
        if args.dataset == "isic":
            net = UNet.UNet(args.num_classes).to(args.device)
    assert net is not None
    utils.log("Network", {"net": net})

    # 初始化数据集
    iniDataSet = InitDataSet(args, dataset_path="./data")
    if args.dataset == "mnist":
        iniDataSet.addTrans(transforms.Normalize((0.1307,), (0.3081,)))  # mnist
    elif args.dataset == "cifar10":
        iniDataSet.addTrans(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # cifar
    elif args.dataset == "isic":
        iniDataSet.addTrans(transform=transforms.Resize(512))

    # 初始化训练类
    if args.iid:
        # Fed i.i.d
        trainer = GlobalTrain(args, iniDataSet)
    else:
        # non-Fed
        trainer = Train(args, iniDataSet)

    # loss, eval_res, best_acc, best_net = trainer.train(net)  # mnist ,cifar
    loss, eval_res, best_acc, best_net = trainer.train(net, nn.BCEWithLogitsLoss())  # isic

    utils.log("Best_Acc:", {"Acc": best_acc})
    utils.save_model(best_net, save_name=name, save_path="./save/pt", full=False)
    utils.draw(args, eval_res, "Epoch", "Acc-Loss", save=True, save_name=name, save_path="./save/png")
