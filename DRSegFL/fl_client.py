#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import logging
import os
import os.path as osp

import numpy as np
from socketIO_client import SocketIO

from DRSegFL import utils
from models.Models import Models

logging.getLogger("client").setLevel(logging.WARNING)
log_dir = osp.join(osp.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)


class LocalModel(object):
    def __init__(self, config: dict, logger):
        self.config = config
        self.model = getattr(Models, self.config["model_name"])(config, logger)

    def train(self):
        local_ep = self.config["epoch"]
        losses = self.model.train(local_ep)
        return self.model.get_weights(), np.mean(losses)

    def eval(self):
        loss, acc = self.model.eval()
        return loss, acc

    def test(self):
        loss, acc = self.model.test()
        return loss, acc


class FederatedClient(object):
    def __init__(self, server_host, server_port, client_config: str):
        self.client_config = utils.load_json(client_config)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.client_config["gpu"]

        self.logger = logging.getLogger("client")
        fh = logging.FileHandler(osp.join(log_dir), self.client_config["log_filename"])
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.ERROR)
        sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(sh)

        self.logger.info(self.client_config)

        self.local_model = None

        self.socketio = SocketIO(server_host, server_port, None, {"timeout": 60000})
        self.register_handles()
        self.socketio.emit("client_wake_up")
        self.socketio.wait()

    def register_handles(self):
        def on_connect():
            logging.info("connect")

        def close_connect():
            logging.info("close connect")

        def re_connect():
            logging.info("re connect")

        def on_init():
            pass
