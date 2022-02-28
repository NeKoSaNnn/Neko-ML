#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
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
            net = UNet.UNet(args.num_channels, args.num_classes).to(args.device)
    assert net
    utils.log("Network", {"net": net}, is_print=False)

    # 初始化数据集
    if args.dataset == "mnist":
        # mnist
        iniDataSet = InitDataSet(args, dataset_path="./data",
                                 trans=transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize((0.1307,), (0.3081,))]))
    elif args.dataset == "cifar10":
        # cifar
        iniDataSet = InitDataSet(args, dataset_path="./data",
                                 trans=transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    elif args.dataset == "isic":
        # isic
        iniDataSet = InitDataSet(args, dataset_path="./data", trans=transforms.ToTensor(),
                                 transTarget=transforms.ToTensor())
    else:
        iniDataSet = None

    assert iniDataSet
    # 初始化训练类
    if args.iid:
        # Fed i.i.d
        trainer = GlobalTrain(args, iniDataSet)
        train_eval, val_eval, test_eval = trainer.train(net, is_eval=True)  # isic
    else:
        # non-Fed
        trainer = Train(args, iniDataSet)
        # train_eval, val_eval, test_eval = trainer.trainMultiClassifier(net)  # mnist ,cifar
        train_eval, val_eval, test_eval = trainer.trainSegmentation(net, is_eval=True)  # isic

    # utils.log("Best_Acc:", {"Train": train_eval.get_best()["acc"], "Val": val_eval.get_best()["acc"],
    #                         "Test": test_eval.get_best()["acc"]})
    # utils.draw(args, eval_res, "Epoch", "Acc-Loss", save=True, save_name=name, save_path="./save/png")

    utils.log("Best_loss:", {"Train": train_eval.get_best()["loss"], "Val": val_eval.get_best()["loss"],
                             "Test": test_eval.get_best()["loss"]})
    train_eval.save_best_model(only_weight=True)
    val_eval.save_best_model(only_weight=True)
    test_eval.save_best_model(only_weight=True)
    loss = {"train": train_eval.get_loss(), "val": val_eval.get_loss(), "test": test_eval.get_loss()}
    acc = {"train": train_eval.get_acc(), "val": val_eval.get_acc(), "test": test_eval.get_acc()}
    utils.draw(args, loss, xLabel="Epoch", yLabel="Loss", save_name="Loss_isic_unet_epoch_{}".format(args.epochs),
               save_path="./save/loss")
    utils.draw(args, acc, xLabel="Epoch", yLabel="Acc", save_name="Acc_isic_unet_epoch_{}".format(args.epochs),
               save_path="./save/acc")
