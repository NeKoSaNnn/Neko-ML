#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
from torchvision import transforms

from datasets import InitDataSet
from models import MLP, CNN
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
        else:
            exit("dataset {} can't be identified".format(args.dataset))
    else:
        exit("model {} can't be identified".format(args.model))
    utils.log("Network", {"net": net})

    # 初始化数据集
    iniDataSet = InitDataSet(args, dataset_path="./data")
    if args.dataset == "mnist":
        iniDataSet.addTrans(transforms.Normalize((0.1307,), (0.3081,)))  # mnist
    elif args.dataset == "cifar10":
        iniDataSet.addTrans(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # cifar

    # 初始化训练类
    if args.iid:
        # Fed i.i.d
        trainer = GlobalTrain(args, iniDataSet)
        loss, eval_acc, best_acc, best_net = trainer.train(net)
    else:
        # non-Fed
        trainer = Train(args, iniDataSet)
        loss, eval_acc, best_acc, best_net = trainer.train(net)

    utils.save_model(best_net, save_name=name, save_path="./save/pt", full=False)
    utils.draw(loss, "Epoch", "Loss", save=True, save_name=name + "-loss", save_path="./save/png")
    print(eval_acc)
    utils.draw(eval_acc, "Eval_Interval", "Acc", save=True, save_name=name + "-acc", save_path="./save/png")
