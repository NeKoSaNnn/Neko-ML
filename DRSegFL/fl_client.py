#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import argparse
import logging
import os
import os.path as osp
import random
import sys

import numpy as np
import socketio

root_dir_name = osp.dirname(sys.path[0])  # ...Neko-ML/
now_dir_name = sys.path[0]  # ...DRSegFL/
sys.path.append(root_dir_name)

from DRSegFL import utils, constants
from DRSegFL.logger import Logger
from DRSegFL.models.Models import Models

DEBUG = True


class LocalModel(object):
    def __init__(self, config: dict, logger):
        self.config = config
        self.local_epoch = self.config[constants.EPOCH]
        self.model = getattr(Models, self.config[constants.NAME_MODEL])(config, logger)
        self.weights_path = self.config[constants.PATH_WEIGHTS]

    def get_weights(self):
        return self.model.get_weights()

    def auto_set_weights(self):
        weights = utils.pickle2obj(self.weights_path)
        self.model.set_weights(weights)

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

    def eval(self, eval_type):
        loss, acc = self.model.eval(eval_type)
        return loss, acc


class FederatedClient(object):
    def __init__(self, client_config_path: str, server_host=None, server_port=None):
        self.client_config = utils.load_json(client_config_path)
        self.server_host = self.client_config[constants.HOST] if server_host is None else server_host
        self.server_port = self.client_config[constants.PORT] if server_port is None else server_port

        os.environ["CUDA_VISIBLE_DEVICES"] = self.client_config["gpu"]

        self.local_epoch = self.client_config[constants.EPOCH]

        self.logger = logging.getLogger(constants.CLIENT)
        log_formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(self.client_config[constants.PATH_LOGFILE])
        fh.setLevel(logging.DEBUG) if DEBUG else fh.setLevel(logging.INFO)
        fh.setFormatter(log_formatter)
        self.logger.addHandler(fh)
        # attention!!!
        # logger has its own level ,default is WARNING
        # set the lowest level to make handle level come into effect
        self.logger.setLevel(logging.DEBUG) if DEBUG else self.logger.setLevel(logging.INFO)

        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING)
        sh.setFormatter(log_formatter)
        self.logger.addHandler(sh)

        self.logger = Logger(self.logger)

        self.logger.info("=" * 100)
        self.logger.info(self.client_config)

        self.local_model = None
        self.ignore_loadavg = self.client_config["ignore_loadavg"]

        # self.socketio = SocketIO(self.server_host, self.server_port, None, {"timeout": 36000})
        self.socketio = socketio.Client()
        self.register_handles()
        self.socketio.connect("http://{}:{}".format(self.server_host, self.server_port))

    def wakeup(self):
        self.logger.info("Client Start {}:{}".format(self.server_host, self.server_port))
        self.socketio.emit("client_wakeup")
        self.socketio.wait()

    def register_handles(self):
        @self.socketio.on("connect")
        def connect():
            self.logger.info("Connect")

        @self.socketio.on("reconnect")
        def reconnect():
            self.logger.info("Re Connect")

        @self.socketio.on("disconnect")
        def disconnect():
            self.logger.info("Close Connect.")

        @self.socketio.on("client_init")
        def client_init():
            self.logger.info("Init ...")
            self.local_model = LocalModel(self.client_config, self.logger)
            self.logger.info("Local Model Init Completed.")

            self.socketio.emit("client_ready")

        @self.socketio.on("client_check_resource")
        def client_check_resource(*args):
            self.logger.info("Start Check Resource ...")
            data = args[0]
            if self.ignore_loadavg:
                self.logger.info("Ignore Loadavg")
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
                self.logger.info("Loadavg : {}".format(loadavg))

            self.socketio.emit("client_check_resource_complete", {"now_global_epoch": data["now_global_epoch"], "loadavg": loadavg})
            self.logger.info("Check Resource Completed.")

        @self.socketio.on("local_update")
        def local_update(*args):
            self.logger.info("Local Update Receiving ...")

            self.logger.debug("args={}".format(args))

            data = args[0]
            sid = data["sid"]

            self.logger.debug("receive_data={}".format(data))
            self.logger.debug("sid=[{}]".format(sid))

            now_global_epoch = data["now_global_epoch"]

            self.logger.info("Local Update Start ...")

            # first global epoch
            if now_global_epoch == 1:
                self.logger.info("Receive Init Weights ...")
                now_weights = utils.pickle2obj(data["now_weights"])
                utils.obj2pickle(now_weights, self.local_model.weights_path)  # init local weights
                self.local_model.set_weights(now_weights)
                self.logger.info("Init Local Weights Completed")

            # train local_epoch
            self.logger.info("GlobalEpoch:{} -- Local Train Start ...".format(now_global_epoch))
            weights, loss = self.local_model.train(self.local_epoch)

            pickle_weights = utils.obj2pickle(weights, self.local_model.weights_path)  # pickle weights path

            emit_data = {
                "now_global_epoch": now_global_epoch,
                "now_weights": pickle_weights,
                constants.TRAIN_LOSS: loss,
                constants.TRAIN_ACC: None,
                constants.TRAIN_CONTRIB: self.local_model.get_contribution(constants.TRAIN),
            }

            self.logger.info(
                "Train with_local_weights -- GlobalEpoch:{} -- Client-sid:[{}] -- AvgLoss:{:.4f}".format(now_global_epoch, sid, loss))

            if constants.TRAIN in data and data[constants.TRAIN]:
                train_loss, train_acc = self.local_model.eval(constants.TRAIN)
                emit_data[constants.TRAIN] = {
                    constants.LOSS: train_loss, constants.ACC: train_acc,
                    constants.CONTRIB: self.local_model.get_contribution(constants.TRAIN)}
                self.logger.info("Eval-Train with_local_weights -- GlobalEpoch:{} -- Client-sid:[{}] --  Loss:{:.4f} , {}".format(
                    now_global_epoch, sid, train_loss, " , ".join(f"{k} : {v:.4f}" for k, v in train_acc.items())))

            if constants.VALIDATION in data and data[constants.VALIDATION]:
                val_loss, val_acc = self.local_model.eval(constants.VALIDATION)
                emit_data[constants.VALIDATION] = {
                    constants.LOSS: val_loss, constants.ACC: val_acc,
                    constants.CONTRIB: self.local_model.get_contribution(constants.VALIDATION)}
                self.logger.info("Eval-Val with_local_weights -- GlobalEpoch:{} -- Client-sid:[{}] --  Loss:{:.4f} , {}".format(
                    now_global_epoch, sid, val_loss, " , ".join(f"{k} : {v:.4f}" for k, v in val_acc.items())))

            if constants.TEST in data and data[constants.TEST]:
                test_loss, test_acc = self.local_model.eval(constants.TEST)
                emit_data[constants.TEST] = {
                    constants.LOSS: test_loss, constants.ACC: test_acc,
                    constants.CONTRIB: self.local_model.get_contribution(constants.TEST)}
                self.logger.info("Eval-Test with_local_weights -- GlobalEpoch:{} -- Client-sid:[{}] -- Loss:{:.4f} , {}".format(
                    now_global_epoch, sid, test_loss, " , ".join(f"{k} : {v:.4f}" for k, v in test_acc.items())))

            self.logger.info("Local Update Complete.")
            self.logger.info("Emit Local Update To Server ...")
            self.socketio.emit("client_update_complete", emit_data)
            self.logger.info("Emit Local Update Completed.")

        @self.socketio.on("eval_with_global_weights")
        def eval_with_global_weights(*args):
            self.logger.info("Receive Global Weights From Server ...")
            data = args[0]
            sid = data["sid"]

            self.logger.debug("receive_data={}".format(data))
            self.logger.debug("sid=[{}]".format(sid))

            now_global_epoch = data["now_global_epoch"]

            global_weights = utils.pickle2obj(data["now_weights"])
            utils.obj2pickle(global_weights, self.local_model.weights_path)  # save global weights to local weights path

            self.local_model.set_weights(global_weights)
            self.logger.info("Update Local Weights Completed.")

            eval_type = data["eval_type"]

            emit_data = {}

            if constants.TRAIN in eval_type:
                train_loss, train_acc = self.local_model.eval(constants.TRAIN)
                self.logger.info(
                    "Train with_global_weights -- GlobalEpoch:{} -- Client-sid:[{}] -- Loss:{:.4f} , {}".format(
                        now_global_epoch, sid, train_loss, " , ".join(f"{k} : {v:.4f}" for k, v in train_acc.items())))
                emit_data[constants.TRAIN] = {
                    constants.LOSS: train_loss, constants.ACC: train_acc,
                    constants.CONTRIB: self.local_model.get_contribution(constants.TRAIN)}

            if constants.VALIDATION in eval_type:
                val_loss, val_acc = self.local_model.eval(constants.VALIDATION)
                self.logger.info(
                    "Val with_global_weights -- GlobalEpoch:{} -- Client-sid:[{}] -- Loss:{:.4f} , {}".format(
                        now_global_epoch, sid, val_loss, " , ".join(f"{k} : {v:.4f}" for k, v in val_acc.items())))
                emit_data[constants.VALIDATION] = {
                    constants.LOSS: val_loss, constants.ACC: val_acc,
                    constants.CONTRIB: self.local_model.get_contribution(constants.VALIDATION)}

            if constants.TEST in eval_type:
                test_loss, test_acc = self.local_model.eval(constants.TEST)
                self.logger.info(
                    "Test with_global_weights -- GlobalEpoch:{} -- Client-sid:[{}] -- Loss:{:.4f} , {}".format(
                        now_global_epoch, sid, test_loss, " , ".join(f"{k} : {v:.4f}" for k, v in test_acc.items())))
                emit_data[constants.TEST] = {
                    constants.LOSS: test_loss, constants.ACC: test_acc,
                    constants.CONTRIB: self.local_model.get_contribution(constants.TEST)}

            self.socketio.emit("eval_with_global_weights_complete", emit_data)

        @self.socketio.on("fin")
        def fin(*args):
            data = args[0]
            if data[constants.FIN]:
                self.logger.info("Federated Learning Client Fin.")
                self.socketio.emit("client_fin", {"sid": data["sid"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_config_path", type=str, required=True, help="path of client config")
    parser.add_argument("--host", type=str, help="optional server host , 'configs/base_config.yaml' has inited host")
    parser.add_argument("--port", type=int, help="optional server port , 'configs/base_config.yaml' has inited port")

    args = parser.parse_args()

    assert osp.exists(args.client_config_path), "{} not exist".format(args.client_config_path)

    try:
        client = FederatedClient(args.client_config_path, args.host, args.port)
        client.wakeup()
    except ConnectionError:
        print("client connect to server error")
