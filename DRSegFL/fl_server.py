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
from typing import List

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
        self.now_global_epoch = 0
        self.now_tolerate = 0

        self.weights_path = self.config[constants.PATH_WEIGHTS]
        self.best_weights_path = self.config[constants.PATH_BEST_WEIGHTS]
        self.tolerate = self.config["tolerate"]

        self.global_stats = {constants.TRAIN: {constants.ACC: [], constants.LOSS: []},
                             constants.VALIDATION: {constants.ACC: [], constants.LOSS: []},
                             constants.TEST: {constants.ACC: [], constants.LOSS: []}}
        self.aggre_stats = {constants.TRAIN: {constants.ACC: [], constants.LOSS: []},
                            constants.VALIDATION: {constants.ACC: [], constants.LOSS: []},
                            constants.TEST: {constants.ACC: [], constants.LOSS: []}}

        self.prev = {constants.TRAIN: {constants.LOSS: None, constants.ACC: None},
                     constants.VALIDATION: {constants.LOSS: None, constants.ACC: None},
                     constants.TEST: {constants.LOSS: None, constants.ACC: None}}
        self.best = {constants.TRAIN: {constants.LOSS: None, constants.ACC: None, constants.WEIGHTS: None, constants.EPOCH: 0},
                     constants.VALIDATION: {constants.LOSS: None, constants.ACC: None, constants.WEIGHTS: None, constants.EPOCH: 0},
                     constants.TEST: {constants.LOSS: None, constants.ACC: None, constants.WEIGHTS: None, constants.EPOCH: 0}}

    def update_global_weights(self, clients_weights, clients_contribution):
        self.global_weights = copy.deepcopy(fed.FedAvg(clients_weights, clients_contribution))

    def get_init_weights(self):
        model = getattr(Models, self.config["model_name"])(self.config, self.logger)
        init_weights = copy.deepcopy(model.get_weights())
        self.logger.info("Init Weights Load Completed")
        del model
        return init_weights

    def get_aggre_loss_acc(self, eval_type: str, client_losses: list, client_acc: List[dict], client_contributions: list, is_record: bool):
        assert eval_type in [constants.TRAIN, constants.VALIDATION, constants.TEST], "aggre eval_type:{} error".format(eval_type)
        now_aggre_loss = utils.list_mean(client_losses, client_contributions)
        now_aggre_acc = utils.dict_list_mean(client_acc, client_contributions)

        if is_record:
            self.aggre_stats[eval_type][constants.LOSS].append(now_aggre_loss)
            self.aggre_stats[eval_type][constants.ACC].append(now_aggre_acc)

        return now_aggre_loss, now_aggre_acc

    def get_global_loss_acc(self, eval_type: str, client_losses: list, client_acc: List[dict], client_contributions: list):
        assert eval_type in [constants.TRAIN, constants.VALIDATION, constants.TEST], "global eval_type:{} error".format(eval_type)
        now_global_loss = utils.list_mean(client_losses, client_contributions)
        now_global_acc = utils.dict_list_mean(client_acc, client_contributions)

        self.global_stats[eval_type][constants.LOSS].append(now_global_loss)
        self.global_stats[eval_type][constants.ACC].append(now_global_acc)

        return now_global_loss, now_global_acc

    def get_stats(self):
        return {
            "global_stats": self.global_stats,
            "aggre_stats": self.aggre_stats
        }

    def update_tolerate(self):
        self.logger.debug("tolerate:{}".format(self.tolerate))
        assert len(self.tolerate.keys()) == 1, "tolerate parameter must just have one"
        tolerate_type = list(self.tolerate.keys())[0]
        assert tolerate_type in [constants.TRAIN, constants.VALIDATION, constants.TEST]
        tolerate_metric = self.tolerate[tolerate_type][constants.METRIC]
        tolerate_num = self.tolerate[tolerate_type][constants.NUM]

        now_stats = {constants.LOSS: self.global_stats[tolerate_type][constants.LOSS][-1],
                     constants.ACC: self.global_stats[tolerate_type][constants.ACC][-1]}
        assert tolerate_metric == constants.LOSS or tolerate_metric in now_stats[constants.ACC].keys(), \
            "metric_tolerate error:{}".format(tolerate_metric)
        if tolerate_metric == constants.LOSS:
            preLoss = self.prev[tolerate_type][constants.LOSS]
            nowLoss = now_stats[constants.LOSS]
            if preLoss and nowLoss > preLoss:
                self.now_tolerate += 1
            else:
                self.now_tolerate = 0
        else:
            preAcc = self.prev[tolerate_type][constants.ACC]
            nowAcc = now_stats[constants.ACC]
            if preAcc and nowAcc[tolerate_metric] < preAcc[tolerate_metric]:
                self.now_tolerate += 1
            else:
                self.now_tolerate = 0

        self.prev[tolerate_type][constants.LOSS] = self.global_stats[tolerate_type][constants.LOSS][-1]
        self.prev[tolerate_type][constants.ACC] = self.global_stats[tolerate_type][constants.ACC][-1]

        if self.now_tolerate >= tolerate_num > 0:
            self.logger.info("{}(metric:{}) Tending To Convergence.".format(tolerate_type, tolerate_metric))
            return True
        return False

    def update_best(self, best_type: str):
        """
        after get_global_loss_acc
        :param best_type:
        :return:
        """
        self.logger.debug("best_type:{}".format(best_type))
        assert self.now_global_epoch == len(self.global_stats[best_type][constants.LOSS])
        assert best_type in [constants.TRAIN, constants.VALIDATION, constants.TEST], "best_type:{} error".format(best_type)
        now_global_loss = self.global_stats[best_type][constants.LOSS][-1]
        now_global_acc = self.global_stats[best_type][constants.ACC][-1]
        global_metric = self.config[constants.GLOBAL_EVAL][best_type]
        if self.best[best_type][constants.ACC] is None:
            self.best[best_type][constants.ACC] = now_global_acc
        if self.best[best_type][constants.LOSS] is None:
            self.best[best_type][constants.LOSS] = now_global_loss
        if (global_metric in now_global_acc.keys() and now_global_acc[global_metric] > self.best[best_type][constants.ACC][global_metric]) \
                or (global_metric == constants.LOSS and now_global_loss < self.best[best_type][constants.LOSS]):
            # default is loss
            self.best[best_type][constants.LOSS] = now_global_loss
            self.best[best_type][constants.ACC] = now_global_acc
            self.best[best_type][constants.WEIGHTS] = copy.deepcopy(self.global_weights)
            self.best[best_type][constants.EPOCH] = self.now_global_epoch

    def save_ckpt(self, save_ckpt_epoch):
        if save_ckpt_epoch is not None:
            if self.now_global_epoch % save_ckpt_epoch == 0:
                ckpt_record_dir = osp.join(osp.dirname(self.weights_path), "record")
                os.makedirs(ckpt_record_dir, exist_ok=True)
                ckpt_record_path = osp.join(ckpt_record_dir, "fed_gep_{}.pth".format(self.now_global_epoch))
                utils.save_weights(self.global_weights, ckpt_record_path)
                self.logger.info("Save Record GlobalWeights -- Epoch : {} -- {}".format(self.now_global_epoch, ckpt_record_path))

    def fin_summary(self, global_eval_types):
        self.logger.info("Federated Learning Summary ...")
        for global_eval_type in global_eval_types:
            assert global_eval_type in [constants.TRAIN, constants.VALIDATION, constants.TEST], \
                "global eval type:{} error".format(global_eval_types)
            now_best = self.best[global_eval_type]
            self.logger.info("Best Global -- {} -- Loss  : {:.4f}".format(global_eval_types, now_best[constants.LOSS]))
            self.logger.info("Best Global -- {} -- Acc   : {}".format(global_eval_types, " , ".
                                                                      join(f"{k} : {v:.4f}" for k, v in now_best[constants.ACC].items())))
            self.logger.info("Best Global -- {} -- Epoch : {}".format(global_eval_types, now_best[constants.EPOCH]))
            if now_best[constants.WEIGHTS]:
                self.logger.info("Save Best GlobalWeights -- {} : {}".format(global_eval_types, self.best_weights_path[global_eval_type]))
                utils.save_weights(now_best[constants.WEIGHTS], self.best_weights_path[global_eval_type])

        self.logger.debug("Global Stats : {}".format(self.get_stats()))


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

        self.logger.info("=" * 100)
        self.logger.info(self.server_config)

        self.NUM_CLIENTS = self.server_config[constants.NUM_CLIENTS]
        self.NUM_GLOBAL_EPOCH = self.server_config[constants.EPOCH]
        self.LOCAL_EVAL = self.server_config[constants.LOCAL_EVAL]
        self.GLOBAL_EVAL = self.server_config[constants.GLOBAL_EVAL]
        self.CLIENT_SINGLE_MAX_LOADAVG = self.server_config["per_client_max_loadavg"]
        self.SAVE_CKPT_EPOCH = self.server_config["save_ckpt_epoch"] if "save_ckpt_epoch" in self.server_config.keys() else None

        self.global_model = GlobalModel(self.server_config, self.logger)

        self.wait_time = 0
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
            return json.dumps(self.global_model.get_stats())

    def clients_check_resource(self):
        self.client_resource = dict()
        if self.fin:
            for sid in self.ready_client_sids:
                emit("fin", {constants.FIN: self.fin, "sid": sid}, room=sid)
        else:
            check_client_sids = np.random.choice(list(self.ready_client_sids), self.NUM_CLIENTS, replace=False)
            for sid in check_client_sids:
                emit("client_check_resource", {"now_global_epoch": self.global_model.now_global_epoch}, room=sid)

    def global_train_next_epoch(self, runnable_client_sids):
        self.global_model.now_global_epoch += 1
        self.client_update_datas = []
        self.logger.info("GlobalEpoch : {}".format(self.global_model.now_global_epoch))
        self.logger.info("Clients-sids : [{}]".format(",".join(runnable_client_sids)))
        now_weights_pickle = utils.obj2pickle(self.global_model.global_weights, self.global_model.weights_path)

        emit_data = {"now_global_epoch": self.global_model.now_global_epoch}
        # first global epoch
        if self.global_model.now_global_epoch == 1:
            emit_data["now_weights"] = now_weights_pickle
        else:
            if constants.TRAIN in self.LOCAL_EVAL:
                emit_data[constants.TRAIN] = self.global_model.now_global_epoch % self.LOCAL_EVAL[constants.TRAIN] == 0
            if constants.VALIDATION in self.LOCAL_EVAL:
                emit_data[constants.VALIDATION] = self.global_model.now_global_epoch % self.LOCAL_EVAL[constants.VALIDATION] == 0
            if constants.TEST in self.LOCAL_EVAL:
                emit_data[constants.TEST] = self.global_model.now_global_epoch % self.LOCAL_EVAL[constants.TEST] == 0

        for sid in runnable_client_sids:
            # first global epoch
            if self.global_model.now_global_epoch == 1:
                self.logger.info("First GlobalEpoch , Send Init Weights To Client-sid:[{}]".format(sid))
            emit_data["sid"] = sid
            emit("local_update", emit_data, room=sid)

    def _local_eval(self, eval_type):
        assert eval_type in [constants.TRAIN, constants.VALIDATION, constants.TEST], "eval_type:{} error".format(eval_type)
        avg_loss, avg_acc = self.global_model.get_aggre_loss_acc(
            eval_type,
            [client_data[eval_type][constants.LOSS] for client_data in self.client_update_datas],
            [client_data[eval_type][constants.ACC] for client_data in self.client_update_datas],
            [client_data[eval_type][constants.CONTRIB] for client_data in self.client_update_datas],
            True)

        self.logger.info("Eval-{} with_locals_weights -- GlobalEpoch:{} -- AvgLoss:{:.4f} , Avg : {}".format(
            eval_type, self.global_model.now_global_epoch, avg_loss,
            " , ".join(f"{k} : {v:.4f}" for k, v in avg_acc.items())))

    def _global_eval(self, eval_type):
        assert eval_type in [constants.TRAIN, constants.VALIDATION, constants.TEST], "eval_type:{} error".format(eval_type)
        global_loss, global_acc = self.global_model.get_global_loss_acc(
            eval_type,
            [client_data[eval_type][constants.LOSS] for client_data in self.client_eval_datas],
            [client_data[eval_type][constants.ACC] for client_data in self.client_eval_datas],
            [client_data[eval_type][constants.CONTRIB] for client_data in self.client_eval_datas])

        self.logger.info("Eval-{} with_global_weights -- GlobalEpoch:{} -- Loss:{:.4f} , {}".format(
            eval_type, self.global_model.now_global_epoch, global_loss,
            " , ".join(f"{k} : {v:.4f}" for k, v in global_acc.items())))

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

        @self.socketio.on("client_wakeup")
        def client_wakeup_handle():
            self.logger.info("[{}] Wake Up".format(request.sid))
            emit("client_init")

        @self.socketio.on("client_ready")
        def client_ready_handle():
            self.logger.info("Client-sid:[{}] Ready For Training".format(request.sid))
            self.ready_client_sids.add(request.sid)
            self.logger.info("Now {} Client(s) Ready ...".format(len(self.ready_client_sids)))
            if len(self.ready_client_sids) >= self.NUM_CLIENTS and self.global_model.now_global_epoch == 0:
                self.logger.info(
                    "Now Ready Client(s) >= {}(num_clients) , Federated Train Start ~".format(self.NUM_CLIENTS))
                self.clients_check_resource()
            # elif len(self.ready_client_sids) < self.NUM_CLIENTS:
            #     self.logger.info(
            #         "{} Client(s) Ready, Waiting Enough Clients To Run...".format(len(self.ready_client_sids)))
            # else:
            #     self.logger.error("Now GlobalEpoch != 0 , Please Restart Server")

        @self.socketio.on("client_check_resource_complete")
        def client_check_resource_complete_handle(data):
            if data["now_global_epoch"] == self.global_model.now_global_epoch:
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

            if self.global_model.now_global_epoch == data["now_global_epoch"]:
                data["now_weights"] = copy.deepcopy(utils.pickle2obj(data["now_weights"]))
                self.client_update_datas.append(data)
                # all clients upload complete
                if self.NUM_CLIENTS == len(self.client_update_datas):
                    local_eval_types = list(self.client_update_datas[0].keys())

                    self.global_model.update_global_weights(
                        [client_data["now_weights"] for client_data in self.client_update_datas],
                        [client_data[constants.TRAIN_CONTRIB] for client_data in self.client_update_datas])

                    global_train_loss, _ = self.global_model.get_aggre_loss_acc(
                        constants.TRAIN,
                        [client_data[constants.TRAIN_LOSS] for client_data in self.client_update_datas],
                        None,
                        [client_data[constants.TRAIN_CONTRIB] for client_data in self.client_update_datas],
                        False)

                    self.logger.info(
                        "Train -- GlobalEpoch:{} -- AvgLoss:{:.4f}".format(self.global_model.now_global_epoch, global_train_loss))

                    if constants.TRAIN in local_eval_types:
                        self._local_eval(constants.TRAIN)

                    if constants.VALIDATION in local_eval_types:
                        self._local_eval(constants.VALIDATION)

                    if constants.TEST in local_eval_types:
                        self._local_eval(constants.TEST)

                    now_weights_pickle = utils.obj2pickle(self.global_model.global_weights,
                                                          self.global_model.weights_path)  # weights path
                    emit_data = {"now_global_epoch": self.global_model.now_global_epoch,
                                 "now_weights": now_weights_pickle,
                                 "eval_type": list(self.GLOBAL_EVAL.keys())}

                    self.client_eval_datas = []  # empty eval datas for next eval epoch
                    for sid in self.ready_client_sids:
                        emit_data["sid"] = sid
                        emit("eval_with_global_weights", emit_data, room=sid)
                        self.logger.info("Server Send Federated Weights To Client-sid:[{}]".format(sid))

        @self.socketio.on("eval_with_global_weights_complete")
        def eval_with_global_weights_complete_handle(data):
            self.logger.info("Receive Client-sid:[{}] Eval Datas:{}".format(request.sid, data))

            self.client_eval_datas.append(data)

            if len(self.client_eval_datas) == self.NUM_CLIENTS:
                global_eval_types = list(self.client_eval_datas[0].keys())

                if constants.TRAIN in global_eval_types:
                    self._global_eval(constants.TRAIN)
                    self.global_model.update_best(constants.TRAIN)

                if constants.VALIDATION in global_eval_types:
                    self._global_eval(constants.VALIDATION)
                    self.global_model.update_best(constants.VALIDATION)

                if constants.TEST in global_eval_types:
                    self._global_eval(constants.TEST)
                    self.global_model.update_best(constants.TEST)

                self.fin = self.global_model.update_tolerate()

                self.global_model.save_ckpt(self.SAVE_CKPT_EPOCH)

                if self.global_model.now_global_epoch >= self.NUM_GLOBAL_EPOCH > 0:
                    self.fin = True
                    self.logger.info("Go to NUM_GLOBAL_EPOCH:{}".format(self.NUM_GLOBAL_EPOCH))

                if not self.fin:
                    # next global epoch
                    self.logger.info("Start Next Global-Epoch Training ...")
                else:
                    self.global_model.fin_summary(global_eval_types)

                self.clients_check_resource()

        @self.socketio.on("client_fin")
        def handle_client_fin(data):
            sid = data["sid"]
            self.fin_client_sids.add(sid)
            self.logger.info("Federated Learning Client-sid:[{}] Fin.".format(sid))
            disconnect(sid)
            if sid in self.ready_client_sids:
                self.ready_client_sids.remove(request.sid)
            if len(self.ready_client_sids) == 0:
                self.logger.info("All Clients Fin. Federated Learning Server Fin.")
                self.socketio.stop()


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
