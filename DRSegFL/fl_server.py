#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import argparse
import copy
import json
import logging
import math
import os
import os.path as osp
import sys
import time

import numpy as np
from flask import Flask, request, render_template
from flask_socketio import SocketIO, emit, disconnect

root_dir_name = osp.dirname(sys.path[0])  # ...Neko-ML/
now_dir_name = sys.path[0]  # ...DRSegFL/
sys.path.append(root_dir_name)

from DRSegFL import utils, constants, fed
from DRSegFL.models.Models import Models

DEBUG = True


class GlobalModel(object):
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.global_weights = self.get_init_weights()
        self.weights_path = self.config[constants.PATH_WEIGHTS]
        self.best_weights_path = self.config[constants.PATH_BEST_WEIGHTS]

        self.global_train_loss = []
        self.global_train_acc = []
        self.global_val_loss = []
        self.global_val_acc = []
        self.global_test_loss = []
        self.global_test_acc = []

        self.prev_val_loss = None
        self.best_val_loss = math.inf
        self.best_val_acc = 0
        self.best_val_weights = None
        self.best_val_global_epoch = 0

        self.prev_test_loss = None
        self.best_test_loss = math.inf
        self.best_test_acc = 0
        self.best_test_weights = None
        self.best_test_global_epoch = 0

    def update_global_weights(self, clients_weights, clients_contribution):
        self.global_weights = fed.FedAvg(clients_weights, clients_contribution)

    def get_init_weights(self):
        model = getattr(Models, self.config["model_name"])(self.config, self.logger)
        init_weights = copy.deepcopy(model.get_weights())
        self.logger.info("Init Weights Load Completed")
        del model
        return init_weights

    def get_global_loss_acc(self, now_global_epoch: int, eval_type: str, client_losses: list, client_acc: list or None,
                            client_contributions: list):
        total_contributions = sum(client_contributions)

        now_global_loss = sum(client_losses[i] * (client_contributions[i] / total_contributions) for i in
                              range(len(client_contributions))) if client_losses else None

        now_global_acc = sum(client_acc[i] * (client_contributions[i] / total_contributions) for i in
                             range(len(client_contributions))) if client_acc else None

        if eval_type == constants.TRAIN:
            self.global_train_loss.append([now_global_epoch, now_global_loss])
            self.global_train_acc.append([now_global_epoch, now_global_acc])
        elif eval_type == constants.VALIDATION:
            self.global_val_loss.append([now_global_epoch, now_global_loss])
            self.global_val_acc.append([now_global_epoch, now_global_acc])
        elif eval_type == constants.TEST:
            self.global_test_loss.append([now_global_epoch, now_global_loss])
            self.global_test_acc.append([now_global_epoch, now_global_acc])
        else:
            self.logger.error("Get Eval Loss And Acc Error ! Error Eval_type :{}".format(eval_type))
        return now_global_loss, now_global_acc

    def get_global_stats(self):
        return {
            "global_train_loss": self.global_train_loss,
            "global_val_loss": self.global_val_loss,
            "global_val_acc": self.global_val_acc,
            "global_test_loss": self.global_test_loss,
            "global_test_acc": self.global_test_acc,
        }


class FederatedServer(object):
    def __init__(self, server_config_path: str, host=None, port=None):
        self.server_config = utils.load_json(server_config_path)
        self.server_host = self.server_config[constants.HOST] if host is None else host
        self.server_port = self.server_config[constants.PORT] if port is None else port

        os.environ["CUDA_VISIBLE_DEVICES"] = self.server_config["gpu"]

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        self.logger = logging.getLogger(constants.SERVER)
        log_formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(self.server_config[constants.PATH_LOGFILE])
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

        self.logger.info(self.server_config)

        self.NUM_CLIENTS = self.server_config[constants.NUM_CLIENTS]
        self.NUM_GLOBAL_EPOCH = self.server_config[constants.EPOCH]
        self.NUM_TOLERATE = self.server_config["num_tolerate"]
        self.TYPE_TOLERATE = self.server_config["type_tolerate"]
        self.EVAL = self.server_config["eval"]
        self.CLIENT_SINGLE_MAX_LOADAVG = self.server_config["per_client_max_loadavg"]

        self.global_model = GlobalModel(self.server_config, self.logger)

        self.now_global_epoch = 0
        self.wait_time = 0
        self.now_tolerate = 0
        self.fin = False
        self.ready_client_sids = set()
        self.fin_client_sids = set()
        self.client_resource = dict()
        self.client_update_datas = []  # now global epoch , all client-update datas
        self.client_eval_datas = []  # now global epoch , all client-eval datas

        self.register_handles()

        @self.app.route("/")
        def home_page():
            return render_template('dashboard.html')

        @self.app.route("/stats")
        def stats_page():
            return json.dumps(self.global_model.get_global_stats())

    def clients_check_resource(self):
        self.client_resource = dict()
        if self.fin:
            for sid in self.ready_client_sids:
                emit("fin", {constants.FIN: self.fin, "sid": sid}, room=sid)
        else:
            check_client_sids = np.random.choice(list(self.ready_client_sids), self.NUM_CLIENTS, replace=False)
            for sid in check_client_sids:
                emit("client_check_resource", {"now_global_epoch": self.now_global_epoch}, room=sid)

    def global_train_next_epoch(self, runnable_client_sids):
        self.now_global_epoch += 1
        self.client_update_datas = []
        self.logger.info("GlobalEpoch : {}".format(self.now_global_epoch))
        self.logger.info("Clients-sids : [{}]".format(",".join(runnable_client_sids)))
        now_weights_pickle = utils.obj2pickle(self.global_model.global_weights, self.global_model.weights_path)

        emit_data = {"now_global_epoch": self.now_global_epoch}
        # first global epoch
        if self.now_global_epoch == 1:
            emit_data["now_weights"] = now_weights_pickle
        else:
            if constants.VALIDATION in self.EVAL:
                emit_data[constants.VALIDATION] = self.now_global_epoch % self.EVAL[constants.VALIDATION] == 0
            if constants.TEST in self.EVAL:
                emit_data[constants.TEST] = self.now_global_epoch % self.EVAL[constants.TEST] == 0

        for sid in runnable_client_sids:
            # first global epoch
            if self.now_global_epoch == 1:
                self.logger.info("First GlobalEpoch , Send Init Weights To Client-sid:[{}]".format(sid))
            emit_data["sid"] = sid
            emit("local_update", emit_data, room=sid)

    def start(self):
        self.logger.info("Server Start {}:{}".format(self.server_host, self.server_port))
        self.socketio.run(self.app, host=self.server_host, port=self.server_port)

    def register_handles(self):
        @self.socketio.on("connect")
        def connect_handle():
            self.logger.info("[{}] Connect".format(request.sid))

        @self.socketio.on("reconnect")
        def reconnect_handle():
            self.logger.info("[{}] Re Connect".format(request.sid))

        @self.socketio.on("disconnect")
        def disconnect_handle():
            self.logger.info("[{}] Close Connect.".format(request.sid))
            if request.sid in self.ready_client_sids:
                self.ready_client_sids.remove(request.sid)

        @self.socketio.on("client_wakeup")
        def client_wakeup_handle():
            self.logger.info("[{}] Wake Up".format(request.sid))
            emit("client_init")

        @self.socketio.on("client_ready")
        def client_ready_handle():
            self.logger.info("Client-sid:[{}] Ready For Training".format(request.sid))
            self.ready_client_sids.add(request.sid)
            if len(self.ready_client_sids) >= self.NUM_CLIENTS and self.now_global_epoch == 0:
                self.logger.info(
                    "{} Client(s) Ready , Federated Train Start ~".format(len(self.ready_client_sids)))
                self.clients_check_resource()
            elif len(self.ready_client_sids) < self.NUM_CLIENTS:
                self.logger.info(
                    "{} Client(s) Ready, Waiting Enough Clients To Run...".format(len(self.ready_client_sids)))
            else:
                self.logger.error("Now GlobalEpoch != 0 , Please Restart Server")

        @self.socketio.on("client_check_resource_complete")
        def client_check_resource_complete_handle(data):
            if data["now_global_epoch"] == self.now_global_epoch:
                self.client_resource[request.sid] = data["loadavg"]
                # up to NUM_CLIENTS , begin next step
                if len(self.client_resource) == self.NUM_CLIENTS:
                    runnable_client_sids = []
                    for sid, loadavg in self.client_resource.items():
                        self.logger.debug("Client-sid:[{}] , Loadavg : {}".format(sid, loadavg))
                        if float(loadavg) < self.CLIENT_SINGLE_MAX_LOADAVG:
                            runnable_client_sids.append(sid)
                            self.logger.info("Client-sid:[{}] Runnable".format(sid))
                        else:
                            self.logger.warning("Client-sid:[{}] Over-loadavg".format(sid))

                    # over half clients runnable
                    if len(runnable_client_sids) / len(self.client_resource) > 0.5:
                        self.wait_time = min(self.wait_time, 3)
                        time.sleep(self.wait_time)
                        self.global_train_next_epoch(runnable_client_sids)
                    else:
                        self.wait_time += 1 if self.wait_time < 10 else 0
                        time.sleep(self.wait_time)
                        self.clients_check_resource()

        @self.socketio.on("client_update_complete")
        def client_update_complete_handle(data):
            self.logger.debug("Received Client-sid:[{}] Update-Data:{} ".format(request.sid, data))

            if self.now_global_epoch == data["now_global_epoch"]:
                data["now_weights"] = copy.deepcopy(utils.pickle2obj(data["now_weights"]))
                self.client_update_datas.append(data)
                # all clients upload complete
                if self.NUM_CLIENTS == len(self.client_update_datas):
                    self.global_model.update_global_weights(
                        [client_data["now_weights"] for client_data in self.client_update_datas],
                        [client_data[constants.TRAIN_CONTRIB] for client_data in self.client_update_datas])

                    global_train_loss, _ = self.global_model.get_global_loss_acc(
                        self.now_global_epoch, constants.TRAIN,
                        [client_data[constants.TRAIN_LOSS] for client_data in self.client_update_datas],
                        None,
                        [client_data[constants.TRAIN_CONTRIB] for client_data in self.client_update_datas])

                    self.logger.info(
                        "Train -- GlobalEpoch:{} -- AvgLoss:{:.4f}".format(self.now_global_epoch,
                                                                           global_train_loss))

                    if constants.VALIDATION_LOSS in self.client_update_datas[0]:
                        avg_val_loss, avg_val_acc = self.global_model.get_global_loss_acc(
                            self.now_global_epoch, constants.VALIDATION,
                            [client_data[constants.VALIDATION_LOSS] for client_data in self.client_update_datas],
                            [client_data[constants.VALIDATION_ACC] for client_data in self.client_update_datas],
                            [client_data[constants.VALIDATION_CONTRIB] for client_data in self.client_update_datas])

                        self.logger.info(
                            "Val with_locals_weights -- GlobalEpoch:{} -- AvgLoss:{:.4f} , AvgAcc:{:.3f}".format(
                                self.now_global_epoch, avg_val_loss, avg_val_acc))

                    if constants.TEST_LOSS in self.client_update_datas[0]:
                        avg_test_loss, avg_test_acc = self.global_model.get_global_loss_acc(
                            self.now_global_epoch, constants.TEST,
                            [client_data[constants.TEST_LOSS] for client_data in self.client_update_datas],
                            [client_data[constants.TEST_ACC] for client_data in self.client_update_datas],
                            [client_data[constants.TEST_CONTRIB] for client_data in self.client_update_datas])

                        self.logger.info(
                            "Test with_locals_weights -- GlobalEpoch:{} -- AvgLoss:{:.4f} , AvgAcc:{:.3f}".format(
                                self.now_global_epoch, avg_test_loss, avg_test_acc))

                    now_weights_pickle = utils.obj2pickle(self.global_model.global_weights,
                                                          self.global_model.weights_path)  # weights path
                    emit_data = {"now_global_epoch": self.now_global_epoch,
                                 "now_weights": now_weights_pickle,
                                 "eval_type": list(self.EVAL.keys())}

                    self.client_eval_datas = []  # empty eval datas for next eval epoch
                    for sid in self.ready_client_sids:
                        emit_data["sid"] = sid
                        emit("eval_with_global_weights", emit_data, room=sid)
                        self.logger.info("Server Send Federated Weights To Client-sid:[{}]".format(sid))

        @self.socketio.on("eval_with_global_weights_complete")
        def eval_with_global_weights_complete_handle(data):
            self.logger.info("Receive Client-sid:[{}] Eval Datas".format(request.sid))

            self.client_eval_datas.append(data)

            if len(self.client_eval_datas) == self.NUM_CLIENTS:
                if constants.VALIDATION in self.client_eval_datas[0]:
                    global_val_loss, global_val_acc = self.global_model.get_global_loss_acc(
                        self.now_global_epoch, constants.VALIDATION,
                        [client_data[constants.VALIDATION][constants.LOSS] for client_data in
                         self.client_eval_datas],
                        [client_data[constants.VALIDATION][constants.ACC] for client_data in
                         self.client_eval_datas],
                        [client_data[constants.VALIDATION][constants.CONTRIB] for client_data in
                         self.client_eval_datas])
                    self.logger.info(
                        "Val with_global_weights -- GlobalEpoch:{} -- AvgLoss:{:.4f} , AvgAcc:{:.3f}".format(
                            self.now_global_epoch, global_val_loss, global_val_acc))

                    if self.TYPE_TOLERATE == constants.VALIDATION:
                        if self.global_model.prev_val_loss and global_val_loss > self.global_model.prev_val_loss:
                            self.now_tolerate += 1
                        else:
                            self.now_tolerate = 0
                    self.global_model.prev_val_loss = global_val_loss

                    if self.now_tolerate > self.NUM_TOLERATE > 0:
                        self.fin = True
                        self.logger.info("Val Tending To Convergence")

                    # Get Best according to Acc
                    if global_val_acc > self.global_model.best_val_acc:
                        self.global_model.best_val_loss = global_val_loss
                        self.global_model.best_val_acc = global_val_acc
                        self.global_model.best_val_weights = self.global_model.global_weights
                        self.global_model.best_val_global_epoch = self.now_global_epoch

                if constants.TEST in self.client_eval_datas[0]:
                    global_test_loss, global_test_acc = self.global_model.get_global_loss_acc(
                        self.now_global_epoch, constants.TEST,
                        [client_data[constants.TEST][constants.LOSS] for client_data in self.client_eval_datas],
                        [client_data[constants.TEST][constants.ACC] for client_data in self.client_eval_datas],
                        [client_data[constants.TEST][constants.CONTRIB] for client_data in self.client_eval_datas])
                    self.logger.info(
                        "Test with_global_weights -- GlobalEpoch:{} -- AvgLoss:{:.4f} , AvgAcc:{:.3f}".format(
                            self.now_global_epoch, global_test_loss, global_test_acc))

                    if self.TYPE_TOLERATE == constants.TEST:
                        if self.global_model.prev_test_loss and global_test_loss > self.global_model.prev_test_loss:
                            self.now_tolerate += 1
                        else:
                            self.now_tolerate = 0
                    self.global_model.prev_test_loss = global_test_loss

                    if self.now_tolerate > self.NUM_TOLERATE > 0:
                        self.fin = True
                        self.logger.info("Test Tending To Convergence")

                    # Get Best according to Acc
                    if global_test_acc > self.global_model.best_test_acc:
                        self.global_model.best_test_loss = global_test_loss
                        self.global_model.best_test_acc = global_test_acc
                        self.global_model.best_test_weights = copy.deepcopy(self.global_model.global_weights)
                        self.global_model.best_test_global_epoch = self.now_global_epoch

                if self.now_global_epoch >= self.NUM_GLOBAL_EPOCH > 0:
                    self.fin = True
                    self.logger.info("Go to NUM_GLOBAL_EPOCH:{}".format(self.NUM_GLOBAL_EPOCH))

                if not self.fin:
                    # next global epoch
                    self.logger.info("Start Next Global-Epoch Training ...")
                else:
                    self.logger.info("Federated Learning Summary ...")
                    self.logger.info("Best Global -- Val -- Loss : {:.4f}".format(self.global_model.best_val_loss))
                    self.logger.info("Best Global -- Val -- Acc  : {:.4f}".format(self.global_model.best_val_acc))
                    self.logger.info(
                        "Best Global -- Val -- Epoch : {}".format(self.global_model.best_val_global_epoch))
                    self.logger.info("Best Global -- Test -- Loss : {:.4f}".format(self.global_model.best_test_loss))
                    self.logger.info("Best Global -- Test -- Acc  : {:.4f}".format(self.global_model.best_test_acc))
                    self.logger.info(
                        "Best Global -- Test -- Epoch : {}".format(self.global_model.best_test_global_epoch))
                    if self.global_model.best_val_weights:
                        self.logger.info("Save Best GlobalWeights -- Val : {}".format(
                            self.global_model.best_weights_path[constants.VALIDATION]))
                        utils.save_weights(self.global_model.best_val_weights,
                                           self.global_model.best_weights_path[constants.VALIDATION])
                    if self.global_model.best_test_weights:
                        self.logger.info("Save Best GlobalWeights -- Test : {}".format(
                            self.global_model.best_weights_path[constants.TEST]))
                        utils.save_weights(self.global_model.best_test_weights,
                                           self.global_model.best_weights_path[constants.TEST])
                self.clients_check_resource()

        @self.socketio.on("client_fin")
        def handle_client_fin(data):
            sid = data["sid"]
            self.fin_client_sids.add(sid)
            disconnect(sid)
            self.logger.info("Federated Learning Client-sid:[{}] Fin.".format(sid))
            if len(list(self.fin_client_sids)) == len(self.NUM_CLIENTS):
                self.logger.info("All Clients Fin . Federated Learning Server Fin.")
                self.socketio.stop()
                exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_config_path", type=str, required=True, help="path of server config")
    parser.add_argument("--host", type=str, help="optional server host , 'configs/base_config.yaml' has inited host")
    parser.add_argument("--port", type=int, help="optional server port , 'configs/base_config.yaml' has inited port")

    args = parser.parse_args()

    assert osp.exists(args.server_config_path), "{} not exist".format(args.server_config_path)

    try:
        server = FederatedServer(args.server_config_path, args.host, args.port)
        server.start()
    except ConnectionError:
        print("server connect error")
