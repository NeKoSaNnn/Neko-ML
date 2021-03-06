#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import argparse

from torch import device, cuda


class neko_args(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # default arguments
        self.parser.add_argument("--epochs", type=int, default=10, help="rounds of training")
        self.parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
        self.parser.add_argument("--train_bs", type=int, default=128, help="train batch size")
        self.parser.add_argument("--test_bs", type=int, default=128, help="test batch size")
        self.parser.add_argument("--momentum", type=float, default=0.5, help="SGD momentum (default: 0.5)")
        self.parser.add_argument("--weight_decay", type=float, default=0,
                                 help="weight decay (L2 penalty) (default: 0)")

        # model arguments
        self.parser.add_argument("--model", type=str, default="mlp", help="model name")
        self.parser.add_argument("--amp", action="store_true", help="use amp or not")

        # other arguments
        self.parser.add_argument("--dataset", type=str, default="mnist", help="name of dataset")
        self.parser.add_argument("--input_size", type=int, default=28 * 28,
                                 help="the input size of dataset for fc-layer")
        self.parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
        self.parser.add_argument("--num_channels", type=int, default=3, help="number of channels of input imgs")
        self.parser.add_argument("--num_workers", type=int, default=1, help="number of workers")
        self.parser.add_argument("--gpu", type=int, default=0, help="GPU ID, -1 for CPU")
        self.parser.add_argument("--eval_interval", type=int, default=10, help="the interval of train epoch to eval")
        self.parser.add_argument("--stopping_rounds", type=int, default=10, help="rounds of early stopping")
        self.parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
        self.parser.add_argument("--verbose", action="store_true", help="verbose print")
        self.parser.add_argument("--log_interval", type=int, default=10, help="the iteration interval of log print")

    def get(self):
        args = self.parser.parse_args()
        args.device = device("cuda:{}".format(args.gpu) if cuda.is_available() and args.gpu != -1 else "cpu")
        return args
