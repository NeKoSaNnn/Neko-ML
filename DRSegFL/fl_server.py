#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import json
import logging
import os
import os.path as osp
import sys
import time

import numpy as np
from flask import Flask, request
from flask_socketio import SocketIO, emit

import fed
from DRSegFL import utils
from models.Models import Models

log_dir = osp.join(osp.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)


class GlobalModel(object):
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.global_weights = self.get_init_parameter()
        self.weights_path = self.config["weights_path"]

        self.global_train_loss = []
        self.global_train_acc = []
        self.global_val_loss = []
        self.global_val_acc = []
        self.global_test_loss = []
        self.global_test_acc = []

    def update_global_weights(self, clients_weights, clients_contribution):
        self.global_weights = fed.FedAvg(clients_weights, clients_contribution)

    def get_init_parameter(self):
        model = getattr(Models, self.config["model_name"])(self.config, self.logger)
        init_parameters = model.get_weights()
        self.logger.info("init parameter loader complete")
        del model
        return init_parameters

    def get_global_loss_acc(self, now_global_epoch: int, eval_type: str, client_losses: list, client_acc: list or None,
                            client_contributions: list):
        total_contributions = np.sum(client_contributions)

        now_global_loss = np.sum(client_losses[i] * (client_contributions[i] / total_contributions) for i in
                                 range(len(client_contributions))) if client_losses is not None else None

        now_global_acc = np.sum(client_acc[i] * (client_contributions[i] / total_contributions) for i in
                                range(len(client_contributions))) if client_acc is not None else None

        if eval_type == "train":
            self.global_train_loss.append([now_global_epoch, now_global_loss])
            self.global_train_acc.append([now_global_epoch, now_global_acc])
        elif eval_type == "val":
            self.global_val_loss.append([now_global_epoch, now_global_loss])
            self.global_val_acc.append([now_global_epoch, now_global_acc])
        elif eval_type == "test":
            self.global_test_loss.append([now_global_epoch, now_global_loss])
            self.global_test_acc.append([now_global_epoch, now_global_acc])
        else:
            logging.error("get eval loss and acc error ! error eval_type :", eval_type)
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
        def __init__(self, host, port, server_config: str):
            self.server_host = host
            self.server_port = port
            self.server_config = utils.load_json(server_config)
            self.app = Flask(__name__)
            self.socketio = SocketIO(self.app)

            self.logger = logging.getLogger("server")
            fh = logging.FileHandler(self.server_config["logfile_path"])
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)

            sh = logging.StreamHandler()
            sh.setLevel(logging.ERROR)
            sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(sh)

            self.logger.info(self.server_config)

            self.NUM_WORKERS = self.server_config["num_workers"]
            self.NUM_CLIENTS = self.server_config["num_clients"]
            self.NUM_GLOBAL_EPOCH = self.server_config["epoch"]
            self.NUM_TOLERATE = self.server_config["num_tolerate"]
            self.EVAL_EPOCH_INTERVAL = self.server_config["eval_interval_epoch"]
            self.CLIENT_SINGLE_MAX_LOADAVG = self.server_config["per_client_max_loadavg"]

            self.global_model = GlobalModel(self.server_config, self.logger)

            self.now_global_epoch = -1
            self.wait_time = 0
            self.ready_client_sids = set()
            self.client_resource = dict()
            self.client_upload_datas = []  # now global epoch , all client-update upload datas

            self.register_handles()

            @self.app.route("/stats")
            def stats():
                return json.dumps(self.global_model.get_global_stats())

        def clients_check_resource(self):
            self.client_resource = dict()
            check_client_sids = np.random.choice(self.ready_client_sids, self.NUM_CLIENTS, replace=False)
            for sid in check_client_sids:
                emit("client_check_resource", {"now_global_epoch": self.now_global_epoch}, room=sid)

        def global_train_next_epoch(self, runnable_client_sids):
            self.now_global_epoch += 1
            self.client_upload_datas = []
            self.logger.info("Global Epoch : {}".format(self.now_global_epoch))
            self.logger.info("locals update from {}".format(runnable_client_sids))
            now_weights_pickle = utils.obj2pickle(self.global_model.global_weights, self.global_model.weights_path)

            emit_data = {"now_global_epoch": self.now_global_epoch}
            # first global epoch
            if self.now_global_epoch == 0:
                emit_data["now_weights"] = now_weights_pickle
                emit_data["weights_path"] = self.global_model.weights_path
            else:
                if "val" in self.EVAL_EPOCH_INTERVAL:
                    emit_data["val"] = self.EVAL_EPOCH_INTERVAL["val"] % self.now_global_epoch == 0
                if "test" in self.EVAL_EPOCH_INTERVAL:
                    emit_data["test"] = self.EVAL_EPOCH_INTERVAL["test"] % self.now_global_epoch == 0

            for sid in runnable_client_sids:
                if self.now_global_epoch == 0:
                    self.logger.info("first global epoch , send init weights to client-sid:{}".format(sid))
                emit("local_update", emit_data, room=sid)

        def register_handles(self):
            @self.socketio.on("connect")
            def connect_handle():
                self.logger.info(request.sid, "connect")

            @self.socketio.on("reconnect")
            def reconnect_handle():
                self.logger.info(request.sid, "re connect")

            @self.socketio.on("disconnect")
            def disconnect_handle():
                self.logger.info(request.sid, "close connect")
                if request.sid in self.ready_client_sids:
                    self.ready_client_sids.remove(request.sid)

            @self.socketio.on("client_wakeup")
            def client_wakeup_handle():
                self.logger.info(request.sid, "wake up")
                emit("client_init")

            @self.socketio.on("client_ready")
            def client_ready_handle():
                self.logger.info(request.sid, "ready for training")
                self.ready_client_sids.add(request.sid)
                if len(self.ready_client_sids) >= self.NUM_WORKERS and self.now_global_epoch == -1:
                    self.logger.info("{} client(s) ready , federated train start ~".format(len(self.ready_clients)))
                    self.clients_check_resource()
                elif len(self.ready_client_sids) < self.NUM_WORKERS:
                    self.logger.info("now get {} ready client(s) , waiting ...".format(len(self.ready_clients)))
                else:
                    self.logger.error("now global epoch != -1 , please restart server")

            @self.socketio.on("client_check_resource_complete")
            def client_check_resource_complete_handle(data):
                if data["now_global_epoch"] == self.now_global_epoch:
                    self.client_resource[request.sid] = data["loadavg"]
                    # up to NUM_CLIENTS , begin next step
                    if len(self.client_resource) == self.NUM_CLIENTS:
                        runnable_client_sids = []
                        for sid, loadavg in self.client_resource.items():
                            self.logger.info("client-sid : {} loadavg : {}".format(sid, loadavg))
                            if float(loadavg) < self.CLIENT_SINGLE_MAX_LOADAVG:
                                runnable_client_sids.append(sid)
                                self.logger.info("client-sid : {} runnable".format(sid))
                            else:
                                self.logger.info("client-sid : {} over-loadavg".format(sid))

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
                self.logger.info("receive client:{} update data:{} bytes".format(request.sid, sys.getsizeof(data)))

                if self.now_global_epoch == data["now_global_epoch"]:
                    data["now_weights"] = utils.pickle2obj(data["now_weights"])
                    self.client_upload_datas.append(data)
                    # all clients upload complete
                    if self.NUM_CLIENTS == len(self.client_upload_datas):
                        self.global_model.update_global_weights(
                            [client_data["now_weights"] for client_data in self.client_upload_datas],
                            [client_data["contribution"] for client_data in self.client_upload_datas])

                        global_train_loss, global_train_acc = self.global_model.get_global_loss_acc(
                            self.now_global_epoch, "train",
                            [client_data["train_loss"] for client_data in self.client_upload_datas],
                            None,
                            [client_data["contribution"] for client_data in self.client_upload_datas])

                        self.logger.info(
                            "Global Epoch:{} -- Train Loss:{.4f} , Acc:{.3f}".format(self.now_global_epoch,
                                                                                     global_train_loss,
                                                                                     global_train_acc))
                        if "val_loss" in self.client_upload_datas[0]:
                            global_val_loss, global_val_acc = self.global_model.get_global_loss_acc(
                                self.now_global_epoch, "val",
                                [client_data["val_loss"] for client_data in self.client_upload_datas],
                                [client_data["val_acc"] for client_data in self.client_upload_datas],
                                [client_data["contribution"] for client_data in self.client_upload_datas])

                            self.logger.info(
                                "Global Epoch:{} -- Val Loss:{.4f} , Acc:{.3f}".format(self.now_global_epoch,
                                                                                       global_val_loss,
                                                                                       global_val_acc))

                        if "test_loss" in self.client_upload_datas[0]:
                            global_test_loss, global_test_acc = self.global_model.get_global_loss_acc(
                                self.now_global_epoch, "test",
                                [client_data["test_loss"] for client_data in self.client_upload_datas],
                                [client_data["test_acc"] for client_data in self.client_upload_datas],
                                [client_data["contribution"] for client_data in self.client_upload_datas])

                            self.logger.info(
                                "Global Epoch:{} -- Test Loss:{.4f} , Acc:{.3f}".format(self.now_global_epoch,
                                                                                        global_test_loss,
                                                                                        global_test_acc))
