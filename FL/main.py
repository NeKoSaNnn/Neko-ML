#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
from utils.args import args
from train import GlobalTrain
import models

if __name__ == "__main__":
    args = args().get()
    global_net = None
    if args.model == "mlp":
        global_net = models.MLP.MLP(args.input_size, 200, args.num_classes).to(args.device)
    elif args.model == "cnn":
        if args.dataset == "mnist":
            global_net = models.CNN.MnistCNN(args.input_size, 200, args.num_classes).to(args.device)
        elif args.dataset == "cifar10":
            global_net = models.CNN.CifarCNN(args.input_size, 200, args.num_classes).to(args.device)
        else:
            exit("dataset {} can't be identified".format(args.dataset))
    else:
        exit("model {} can't be identified".format(args.model))
    GlobalTrain(args).train(global_net)
