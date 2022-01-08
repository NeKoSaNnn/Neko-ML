#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import os.path as osp
import sys
import matplotlib
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

sys.path.append(osp.dirname(sys.path[0]))
from neko import neko_utils


class utils(neko_utils.neko_utils):
    def __init__(self, log_path):
        super(utils, self).__init__(log_path)

    def draw(self, args, vals, xLabel="x", yLabel="y", save=True, save_name=None, sava_suffix=".png",
             save_path="../save"):
        assert isinstance(vals, list) or isinstance(vals, tuple) or isinstance(vals, dict)
        self.mkdir_nf(save_path)
        plt.figure()
        if isinstance(vals, list) or isinstance(vals, tuple):
            plt.plot(range(len(vals)), vals)
        elif isinstance(vals, dict):
            for k, v in vals.items():
                assert isinstance(v, list) or isinstance(v, tuple)
                if k == "train":
                    plt.subplot(211)
                    plt.plot([i * args.eval_interval for i in range(len(v))], v, color="blue", label="train acc")
                    plt.ylabel("Acc")
                    plt.legend()
                if k == "test":
                    plt.subplot(211)
                    plt.plot([i * args.eval_interval for i in range(len(v))], v, color="green", label="test acc")
                    plt.ylabel("Acc")
                    plt.legend()
                if k == "loss":
                    plt.subplot(212)
                    plt.plot(range(len(v)), v, color="red", label="loss")
                    plt.xlabel(xLabel)
                    plt.ylabel("Loss")
                    plt.legend()
        if save:
            save_name = self.get_now_time() if save_name is None else save_name
            plt.savefig(osp.join(save_path, save_name + sava_suffix))
        plt.show()

    def save_model(self, net, save_name, save_path="../save/pt", full=False):
        self.mkdir_nf(save_path)
        save_name = "{}.pt".format(save_name)
        save_path = osp.join(save_path, save_name)
        torch.save(net if full else net.state_dict(), save_path)
        self.log("Save", {save_path: "success"})
