#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
from torch import nn
from torchvision import transforms

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
        iniDataSet.addTrans(transforms.Resize([256, 256]))  # isic
    # 初始化训练类
    if args.iid:
        # Fed i.i.d
        trainer = GlobalTrain(args, iniDataSet)
    else:
        # non-Fed
        trainer = Train(args, iniDataSet)

    # train_eval, val_eval, test_eval = trainer.train(net)  # mnist ,cifar
    train_eval, val_eval, test_eval = trainer.train(net, is_eval=True, loss_f=nn.BCEWithLogitsLoss())  # isic

    # utils.log("Best_Acc:", {"Train": train_eval.get_best()["acc"], "Val": val_eval.get_best()["acc"],
    #                         "Test": test_eval.get_best()["acc"]})
    utils.log("Best_loss:", {"Train": train_eval.get_best()["loss"], "Val": val_eval.get_best()["loss"],
                             "Test": test_eval.get_best()["loss"]})
    train_eval.save_best_model(only_weight=True)

    # utils.draw(args, eval_res, "Epoch", "Acc-Loss", save=True, save_name=name, save_path="./save/png")
