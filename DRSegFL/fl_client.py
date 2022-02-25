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

from DRSegFL import utils, constants
from models.Models import Models

import constants

logging.getLogger("socketio-client").setLevel(logging.WARNING)
log_dir = osp.join(osp.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)


class LocalModel(object):
    def __init__(self, config: dict, logger):
        self.config = config
        self.local_epoch = self.config[constants.EPOCH]
        self.model = getattr(Models, self.config["model_name"])(config, logger)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, params):
        self.model.set_weights(params)

    def get_contribution(self, contribution_type):
        if contribution_type == constants.TRAIN:
            return self.model.train_contribution
        elif contribution_type == constants.VALIDATION:
            return self.model.val_contribution
        elif contribution_type == constants.TEST:
            return self.model.test_contribution
        else:
            raise TypeError

    def train(self, local_epoch):
        losses = self.model.train(local_epoch)
        return self.get_weights(), np.mean(losses)

    def val(self):
        loss, acc = self.model.eval(constants.VALIDATION)
        return loss, acc

    def test(self):
        loss, acc = self.model.eval(constants.TEST)
        return loss, acc


class FederatedClient(object):
    def __init__(self, server_host, server_port, client_config: str):
        self.client_config = utils.load_json(client_config)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.client_config["gpu"]

        self.local_epoch = self.client_config[constants.EPOCH]

        self.logger = logging.getLogger("client")
        fh = logging.FileHandler(self.client_config["logfile_path"])
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.ERROR)
        sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(sh)

        self.logger.info(self.client_config)

        self.local_model = None
        self.ignore_loadavg = True if self.client_config["ignore_loadavg"] == "true" else False

        self.socketio = SocketIO(server_host, server_port, None, {"timeout": 60000})
        self.register_handles()
        self.socketio.emit("client_wakeup")
        self.socketio.wait()

    def register_handles(self):
        def connect():
            logging.info("connect")

        def reconnect():
            logging.info("reconnect")

        def disconnect():
            logging.info("disconnect")

        def client_init():
            logging.info("on init")
            self.local_model = LocalModel(self.client_config, self.logger)
            logging.info("local model init complete")

            self.socketio.emit("client_ready")

        def client_check_resource(*args):
            logging.info("start check resource ...")
            data = args[0]
            if self.ignore_loadavg:
                logging.info("ignore loadavg")
                loadavg = 0.15
            else:
                loadavg_data = {}
                with open("/proc/loadavg") as f:
                    loadavg_raw_data = f.read().split()
                    loadavg_data["loadavg_1min"] = loadavg_raw_data[0]
                    loadavg_data["loadavg_5min"] = loadavg_raw_data[1]
                    loadavg_data["loadavg_15min"] = loadavg_raw_data[2]
                    loadavg_data["loadavg_rate"] = loadavg_raw_data[3]
                    loadavg_data["last_pid"] = loadavg_raw_data[4]

                loadavg = loadavg_data["loadavg_15min"]
                logging.info("loadavg : ", loadavg)

            self.socketio.emit("client_check_resource_complete", {"now_global_epoch": data["now_global_epoch"],
                                                                  "loadavg": loadavg})

        def local_update(*args):
            logging.info("local update ...")
            data = args[0]
            sid = args[1]["room"]

            now_global_epoch = data["now_global_epoch"]

            if now_global_epoch == 0:
                logging.info("receive init weights")
                now_weights = utils.pickle2obj(data["now_weights"])
                self.local_model.set_weights(now_weights)

            # train local_epoch
            weights, loss = self.local_model.train(self.local_epoch)

            pickle_weights = utils.obj2pickle(weights)

            emit_data = {
                "now_global_epoch": now_global_epoch,
                "now_weights": pickle_weights,
                constants.TRAIN_LOSS: loss,
                constants.TRAIN_CONTRIB: self.local_model.get_contribution(constants.TRAIN),
            }

            self.logger.info(
                "Train with local_weights -- Global Epoch:{} -- Client:{} -- Local Epoch:{} AvgLoss:{.4f}".format(
                    now_global_epoch, sid, self.local_epoch, loss))

            if constants.VALIDATION in data and data[constants.VALIDATION]:
                val_loss, val_acc = self.local_model.val()
                emit_data[constants.VALIDATION_LOSS] = val_loss
                emit_data[constants.VALIDATION_ACC] = val_acc
                emit_data[constants.VALIDATION_CONTRIB] = self.local_model.get_contribution(constants.VALIDATION)
                self.logger.info(
                    "Val with local_weights -- Global Epoch:{} -- Client:{} --  Loss:{.4f} , Acc:{.3f}".format(
                        now_global_epoch, sid, val_loss, val_acc))

            if constants.TEST in data and data[constants.TEST]:
                test_loss, test_acc = self.local_model.test()
                emit_data[constants.TEST_LOSS] = test_loss
                emit_data[constants.TEST_ACC] = test_acc
                emit_data[constants.TEST_CONTRIB] = self.local_model.get_contribution(constants.TEST)
                self.logger.info(
                    "Test with local_weights -- Global Epoch:{} -- Client:{}-- Loss:{.4f} , Acc:{.3f}".format(
                        now_global_epoch, sid, test_loss, test_acc))

            self.logger.info("emit local update to server ...")
            self.socketio.emit("client_update_complete", emit_data)
            self.logger.info("emit local update to server complete")

        def eval_with_global_weights(*args):
            self.logger.info("receive federated weights from server")
            data = args[0]
            sid = args[1]["room"]

            now_global_epoch = data["now_global_epoch"]

            global_weights = utils.pickle2obj(data["now_weights"])
            self.local_model.set_weights(global_weights)
            self.logger.info("set federated weights complete")

            eval_type = data["eval_type"]

            emit_data = {}

            if constants.VALIDATION in eval_type:
                val_loss, val_acc = self.local_model.val()
                self.logger.info(
                    "Val with global_weights -- Global Epoch:{} -- Client:{}--  Loss:{.4f} , Acc:{.3f}".format(
                        now_global_epoch, sid, val_loss, val_acc))
                emit_data[constants.VALIDATION_LOSS] = val_loss
                emit_data[constants.VALIDATION_ACC] = val_acc
                emit_data[constants.VALIDATION_CONTRIB] = self.local_model.get_contribution(constants.VALIDATION)

            if constants.TEST in eval_type:
                test_loss, test_acc = self.local_model.test()
                self.logger.info(
                    "Test with global_weights -- Global Epoch:{} -- Client:{}--  Loss:{.4f} , Acc:{.3f}".format(
                        now_global_epoch, sid, test_loss, test_acc))
                emit_data[constants.TEST_LOSS] = test_loss
                emit_data[constants.TEST_ACC] = test_acc
                emit_data[constants.TEST_CONTRIB] = self.local_model.get_contribution(constants.TEST)

            self.socketio.emit("eval_with_global_weights_complete", emit_data)

            if data[constants.STOP]:
                self.logger.info("federated learning fin.")
                exit(0)
