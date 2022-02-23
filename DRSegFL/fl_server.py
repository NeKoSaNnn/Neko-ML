#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import numpy as np

from models.Models import Models


class GlobalModel(object):
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.global_weight = self.get_init_parameter()
        self.global_train_loss = []

    def get_init_parameter(self):
        model = getattr(Models, self.config["model_name"])(self.config, self.logger)
        init_parameters = model.get_weights()
        self.logger.info("init parameter loader complete")
        del model
        return init_parameters

    def get_train_loss(self, client_losses, client_contributions, now_global_epoch):
        total_contributions = np.sum(client_contributions)
        now_global_loss = np.sum(client_losses[i] * (client_contributions[i] / total_contributions) for i in
                                 range(len(client_contributions)))
        self.global_train_loss.append([now_global_epoch, now_global_loss])
        return now_global_loss


class FederatedServer(object):
    def __init__(self):
        pass
