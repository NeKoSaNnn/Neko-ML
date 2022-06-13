#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import argparse
import copy
import json
import logging
import os
import os.path as osp
import sys
import time
from typing import List

import numpy as np
import rsa
from flask import Flask, request, render_template
from flask_socketio import SocketIO, emit, disconnect
from tensorboardX import SummaryWriter

root_dir_name = osp.dirname(sys.path[0])  # ...Neko-ML/
now_dir_name = sys.path[0]  # ...DRSegFL/
sys.path.append(root_dir_name)

from DRSegFL import utils, constants, fed, endecrypt
from DRSegFL.logger import Logger
from DRSegFL.models.Models import Models

DEBUG = True
sleep_time = 0  # for ui


class GlobalModel(object):
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.global_weights = self.get_init_weights()
        self.now_global_epoch = 0
        self.now_tolerate = 0

        self.weights_path = self.config[constants.PATH_WEIGHTS]
        self.best_weights_path = self.config[constants.PATH_BEST_WEIGHTS]
        self.tolerate = self.config["tolerate"] if "tolerate" in self.config else None

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
        model = getattr(Models, self.config[constants.NAME_MODEL])(self.config, self.logger, only_init_weights=True)
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

    def update_tolerate(self, now_type):
        self.logger.debug("tolerate:{}".format(self.tolerate))
        if self.tolerate is None:
            return None
        assert len(self.tolerate.keys()) == 1, "tolerate parameter must just have one"
        tolerate_type = list(self.tolerate.keys())[0]
        assert tolerate_type in [constants.TRAIN, constants.VALIDATION, constants.TEST]
        if now_type == tolerate_type:
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
        return None

    def update_best(self, best_type: str):
        """
        after get_global_loss_acc
        :param best_type:
        :return:
        """
        self.logger.debug("best_type:{}".format(best_type))
        # assert self.now_global_epoch == len(self.global_stats[best_type][constants.LOSS])
        assert best_type in [constants.TRAIN, constants.VALIDATION, constants.TEST], "best_type:{} error".format(best_type)
        now_global_loss = self.global_stats[best_type][constants.LOSS][-1]
        now_global_acc = self.global_stats[best_type][constants.ACC][-1]
        global_metric = self.config[constants.GLOBAL_EVAL][best_type][constants.METRIC]
        # init
        if self.best[best_type][constants.ACC] is None:
            self.best[best_type][constants.ACC] = now_global_acc
        if self.best[best_type][constants.LOSS] is None:
            self.best[best_type][constants.LOSS] = now_global_loss
        if self.best[best_type][constants.WEIGHTS] is None:
            self.best[best_type][constants.WEIGHTS] = copy.deepcopy(self.global_weights)
        if self.best[best_type][constants.EPOCH] == 0:
            self.best[best_type][constants.EPOCH] = self.now_global_epoch
        # compare with global_eval metric
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
                ckpt_record_path = osp.join(ckpt_record_dir, "fed_gep_{}.pt".format(self.now_global_epoch))
                utils.save_weights(self.global_weights, ckpt_record_path)
                self.logger.info("Save Record GlobalWeights -- Epoch : {} -- {}".format(self.now_global_epoch, ckpt_record_path))

    def fin_summary(self, global_eval_types):
        self.logger.info("Federated Learning Summary ...")
        for global_eval_type in global_eval_types:
            assert global_eval_type in [constants.TRAIN, constants.VALIDATION, constants.TEST], \
                "global eval type:{} error".format(global_eval_types)
            now_best = self.best[global_eval_type]
            self.logger.info("Best Global -- {} -- Loss  : {:.4f}".format(global_eval_type, now_best[constants.LOSS]))
            self.logger.info("Best Global -- {} -- Acc   : {}".format(global_eval_type, " , ".
                                                                      join(f"{k} : {v:.4f}" for k, v in now_best[constants.ACC].items())))
            self.logger.info("Best Global -- {} -- Epoch : {}".format(global_eval_type, now_best[constants.EPOCH]))
            if now_best[constants.WEIGHTS]:
                self.logger.info("Save Best GlobalWeights -- {} : {}".format(global_eval_type, self.best_weights_path[global_eval_type]))
                utils.save_weights(now_best[constants.WEIGHTS], self.best_weights_path[global_eval_type])

        self.logger.debug("Global Stats : {}".format(self.get_stats()))


class FederatedServer(object):
    def __init__(self, server_config_path: str, host=None, port=None):
        self.server_config = utils.load_json(server_config_path)
        self.server_host = self.server_config[constants.HOST] if host is None else host
        self.server_port = self.server_config[constants.PORT] if port is None else port
        self.logfile_path = self.server_config[constants.PATH_LOGFILE]

        tbX_dir = self.server_config[constants.DIR_TBX_LOGFILE]
        self.tbX = SummaryWriter(logdir=tbX_dir)

        os.environ["CUDA_VISIBLE_DEVICES"] = self.server_config["gpu"]

        self.app = Flask(__name__, template_folder=osp.join(now_dir_name, "static", "templates"),
                         static_folder=osp.join(now_dir_name, "static"))
        # async_mode = None
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", ping_timeout=36000, ping_interval=300)

        self.logger = logging.getLogger(constants.SERVER)
        log_formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(self.logfile_path)
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
        self.logger.info(self.server_config)

        self.NUM_CLIENTS = self.server_config[constants.NUM_CLIENTS]
        self.NUM_GLOBAL_EPOCH = self.server_config[constants.EPOCH]
        self.LOCAL_EVAL = self.server_config[constants.LOCAL_EVAL] if constants.LOCAL_EVAL in self.server_config else None
        self.GLOBAL_EVAL = self.server_config[constants.GLOBAL_EVAL]
        self.CLIENT_SINGLE_MAX_LOADAVG = self.server_config["per_client_max_loadavg"]
        self.SAVE_CKPT_EPOCH = self.server_config["save_ckpt_epoch"] if "save_ckpt_epoch" in self.server_config.keys() else None

        self.global_model = GlobalModel(self.server_config, self.logger)
        self.pubkey, self.privkey = rsa.newkeys(512)

        self.wait_time = 0
        self.fin = False
        self.ready_client_sids = set()
        self.running_client_sids = set()
        self.client_resource = dict()
        self.client_update_datas = dict()  # now global epoch , all client-update datas
        self.client_eval_datas = dict()  # now global epoch , all client-eval datas
        self.client_pubkeys = dict()

        self.register_handles()

        @self.app.route("/")
        def home_page():
            return render_template("dashboard.html", async_mode=self.socketio.async_mode)

        @self.app.route("/stats")
        def stats_page():
            return json.dumps(self.global_model.get_stats())

    def rsaEncrypt(self, sid, data, dumps=True):
        """
        rsaEncrypt data
        :param sid: the client context sid
        :param data: the data will encrypt
        :param dumps: default is True , whether data need to serialize before encrypt
        :return:
        """
        if sid not in self.client_pubkeys or self.client_pubkeys[sid] is None:
            retry = 10
            while retry > 0:
                emit("get_client_pubkey", broadcast=True)
                time.sleep(3)
                if sid in self.client_pubkeys and self.client_pubkeys[sid] is not None:
                    break
                retry -= 1
        res_data = endecrypt.rsaEncrypt(self.client_pubkeys[sid], data, dumps)
        return res_data

    def rsaDecrypt(self, data, loads=True):
        """
        rsaDecrypt data
        :param data: the data will decrypt
        :param loads: default is True , whether decrypt data need to deserialize
        :return:
        """
        res_data = endecrypt.rsaDecrypt(self.privkey, data, loads)
        return res_data

    def start(self):
        self.logger.info("Server Start {}:{}".format(self.server_host, self.server_port))
        self.socketio.run(self.app, host=self.server_host, port=self.server_port)
        emit("s_connect", broadcast=True, namespace="/ui")  # for ui

    def clients_check_resource(self):
        self.client_resource = dict()
        if self.fin:
            for sid in self.ready_client_sids:
                emit_data = {constants.FIN: self.fin, "sid": sid}
                emit("fin", self.rsaEncrypt(sid, emit_data), room=sid)
        else:
            self.running_client_sids = set(np.random.choice(list(self.ready_client_sids), self.NUM_CLIENTS, replace=False))
            for sid in self.running_client_sids:
                emit("c_check_resource", {"sid": sid}, broadcast=True, namespace="/ui")  # for ui
                time.sleep(sleep_time)
                emit_data = {"now_global_epoch": self.global_model.now_global_epoch}
                emit("client_check_resource", self.rsaEncrypt(sid, emit_data), room=sid)

    def halfway_client_check_resource(self, sid):
        self.running_client_sids.add(sid)
        emit("c_check_resource", {"sid": sid}, broadcast=True, namespace="/ui")  # for ui
        time.sleep(sleep_time)
        emit_data = {"now_global_epoch": self.global_model.now_global_epoch, "halfway": True}
        emit("client_check_resource", self.rsaEncrypt(sid, emit_data), room=sid)

    def global_train_next_epoch(self, runnable_client_sids):
        self.global_model.now_global_epoch += 1
        self.client_update_datas = dict()
        self.logger.info("GlobalEpoch : {}".format(self.global_model.now_global_epoch))
        self.logger.info("Clients-sids : [{}]".format(",".join(runnable_client_sids)))

        now_weights_pickle = utils.obj2pickle(self.global_model.global_weights, self.global_model.weights_path)
        emit_data = {"now_global_epoch": self.global_model.now_global_epoch, "now_weights": now_weights_pickle}
        if self.LOCAL_EVAL:
            if constants.TRAIN in self.LOCAL_EVAL:
                emit_data[constants.TRAIN] = self.global_model.now_global_epoch % self.LOCAL_EVAL[constants.TRAIN] == 0
            if constants.VALIDATION in self.LOCAL_EVAL:
                emit_data[constants.VALIDATION] = self.global_model.now_global_epoch % self.LOCAL_EVAL[constants.VALIDATION] == 0
            if constants.TEST in self.LOCAL_EVAL:
                emit_data[constants.TEST] = self.global_model.now_global_epoch % self.LOCAL_EVAL[constants.TEST] == 0

        for sid in runnable_client_sids:
            emit_data["sid"] = sid
            emit("c_train", {"sid": sid, "gep": self.global_model.now_global_epoch}, broadcast=True, namespace="/ui")  # for ui
            emit("local_update", self.rsaEncrypt(sid, emit_data), room=sid)

    def halfway_train(self, runnable_client_sid):
        self.logger.info("GlobalEpoch : {}".format(self.global_model.now_global_epoch))
        self.logger.info("Clients-sids : [{}]".format(runnable_client_sid))

        now_weights_pickle = utils.obj2pickle(self.global_model.global_weights, self.global_model.weights_path)
        emit_data = {"now_global_epoch": self.global_model.now_global_epoch, "now_weights": now_weights_pickle}
        if self.LOCAL_EVAL:
            if constants.TRAIN in self.LOCAL_EVAL:
                emit_data[constants.TRAIN] = self.global_model.now_global_epoch % self.LOCAL_EVAL[constants.TRAIN] == 0
            if constants.VALIDATION in self.LOCAL_EVAL:
                emit_data[constants.VALIDATION] = self.global_model.now_global_epoch % self.LOCAL_EVAL[constants.VALIDATION] == 0
            if constants.TEST in self.LOCAL_EVAL:
                emit_data[constants.TEST] = self.global_model.now_global_epoch % self.LOCAL_EVAL[constants.TEST] == 0

        emit_data["sid"] = runnable_client_sid
        emit("c_train", {"sid": runnable_client_sid, "gep": self.global_model.now_global_epoch}, broadcast=True, namespace="/ui")  # for ui
        emit("local_update", self.rsaEncrypt(runnable_client_sid, emit_data), room=runnable_client_sid)

    def _local_eval(self, eval_type):
        assert eval_type in [constants.TRAIN, constants.VALIDATION, constants.TEST], "eval_type:{} error".format(eval_type)
        avg_loss, avg_acc = self.global_model.get_aggre_loss_acc(
            eval_type,
            [client_data[eval_type][constants.LOSS] for client_data in self.client_update_datas.values()],
            [client_data[eval_type][constants.ACC] for client_data in self.client_update_datas.values()],
            [client_data[eval_type][constants.CONTRIB] for client_data in self.client_update_datas.values()],
            True)

        self.logger.info("Eval-{} with_locals_weights -- GlobalEpoch:{} -- AvgLoss:{:.4f} , Avg : {}".format(
            eval_type, self.global_model.now_global_epoch, avg_loss,
            " , ".join(f"{k} : {v:.4f}" for k, v in avg_acc.items())))
        self.tbX.add_scalar("local_eval/{}_loss".format(eval_type), avg_loss, self.global_model.now_global_epoch)
        self.tbX.add_scalars("local_eval/{}_acc".format(eval_type), avg_acc, self.global_model.now_global_epoch)

    def _global_eval(self, eval_type):
        assert eval_type in [constants.TRAIN, constants.VALIDATION, constants.TEST], "eval_type:{} error".format(eval_type)
        global_loss, global_acc = self.global_model.get_global_loss_acc(
            eval_type,
            [client_data[eval_type][constants.LOSS] for client_data in self.client_eval_datas.values()],
            [client_data[eval_type][constants.ACC] for client_data in self.client_eval_datas.values()],
            [client_data[eval_type][constants.CONTRIB] for client_data in self.client_eval_datas.values()])

        self.logger.info("Eval-{} with_global_weights -- GlobalEpoch:{} -- Loss:{:.4f} , {}".format(
            eval_type, self.global_model.now_global_epoch, global_loss,
            " , ".join(f"{k} : {v:.4f}" for k, v in global_acc.items())))
        self.tbX.add_scalar("global_eval/{}_loss".format(eval_type), global_loss, self.global_model.now_global_epoch)
        self.tbX.add_scalars("global_eval/{}_acc".format(eval_type), global_acc, self.global_model.now_global_epoch)

    def register_handles(self):
        @self.socketio.on("connect")
        def connect_handle():
            self.logger.info("[{}] Connect".format(request.sid))
            emit("c_connect", {"sid": request.sid}, broadcast=True, namespace="/ui")  # for ui

        @self.socketio.on("connect", namespace="/ui")
        def ui_connect_handle():
            self.logger.info("ui [{}] Connect".format(request.sid))

        @self.socketio.on("reconnect")
        def reconnect_handle():
            self.logger.info("[{}] Re Connect".format(request.sid))
            emit("c_reconnect", {"sid": request.sid}, broadcast=True, namespace="/ui")  # for ui

        @self.socketio.on("reconnect", namespace="/ui")
        def ui_reconnect_handle():
            self.logger.info("ui [{}] Re Connect".format(request.sid))

        @self.socketio.on("disconnect")
        def disconnect_handle():
            sid = request.sid
            self.logger.info("[{}] Close Connect.".format(sid))
            if sid in self.ready_client_sids:
                self.ready_client_sids.remove(sid)
            if sid in self.running_client_sids:
                self.running_client_sids.remove(sid)
            if sid in self.client_update_datas.keys():
                self.client_update_datas.pop(sid)
            if sid in self.client_eval_datas.keys():
                self.client_eval_datas.pop(sid)
            emit("c_disconnect", {"sid": request.sid}, broadcast=True, namespace="/ui")  # for ui

        @self.socketio.on("disconnect", namespace="/ui")
        def ui_disconnect_handle():
            self.logger.info("ui [{}] Close Connect.".format(request.sid))
            emit("ui_disconnect", namespace="/ui")
            disconnect(request.sid, namespace="/ui")

        @self.socketio.on("heartbeat")
        def heartbeat_handle():
            self.logger.debug("Receive HeartBeat from [{}] , Still Alive".format(request.sid))
            emit("re_heartbeat")

        @self.socketio.on_error()
        def error_handle(e):
            self.logger.error(e)

        @self.socketio.on_error(namespace="/ui")
        def ui_error_handle(e):
            self.logger.error("ui:{}".format(e))

        @self.socketio.on("get_server_pubkey")
        def get_server_pubkey():
            emit_data = {"server_pubkey": {"n": str(self.pubkey.n), "e": str(self.pubkey.e)}}
            emit("server_pubkey", emit_data)

        @self.socketio.on("client_pubkey")
        def client_pubkey(data):
            sid = request.sid
            self.client_pubkeys[sid] = rsa.PublicKey(int(data["client_pubkey"]["n"]), int(data["client_pubkey"]["e"]))

        @self.socketio.on("client_wakeup")
        def client_wakeup_handle(data):
            sid = request.sid
            self.client_pubkeys[sid] = rsa.PublicKey(int(data["client_pubkey"]["n"]), int(data["client_pubkey"]["e"]))
            self.logger.info("[{}] Wake Up".format(sid))
            emit("c_wakeup", {"sid": sid}, broadcast=True, namespace="/ui")  # for ui
            emit_data = {"now_global_epoch": self.global_model.now_global_epoch,
                         "server_pubkey": {"n": str(self.pubkey.n), "e": str(self.pubkey.e)}}
            emit("client_init", emit_data)

        @self.socketio.on("client_ready")
        def client_ready_handle():
            sid = request.sid
            self.logger.info("Client-sid:[{}] Ready For Training".format(sid))
            self.ready_client_sids.add(sid)

            if len(self.ready_client_sids) <= self.NUM_CLIENTS and self.global_model.now_global_epoch > 0:
                self.logger.info(
                    "Now GlobalEpoch:{} , A New Client joining ... , Ready Client(s)_Num:{} <= {}(num_clients) .".format(
                        self.global_model.now_global_epoch, len(self.ready_client_sids), self.NUM_CLIENTS))
                self.halfway_client_check_resource(sid)
            if len(self.ready_client_sids) >= self.NUM_CLIENTS and self.global_model.now_global_epoch == 0:
                self.logger.info(
                    "Now Ready Client(s)_Num:{} >= {}(num_clients) , Federated Train Start ~".format(len(self.ready_client_sids),
                                                                                                     self.NUM_CLIENTS))
                self.clients_check_resource()
            elif len(self.ready_client_sids) < self.NUM_CLIENTS and self.global_model.now_global_epoch == 0:
                self.logger.info(
                    "Now Ready Client(s)_Num:{} < {}(num_clients) , Waiting Enough Clients To Run...".format(len(self.ready_client_sids),
                                                                                                             self.NUM_CLIENTS))
            # 注释原因:client可中途加入
            # else:
            #     self.logger.error("Now GlobalEpoch != 0 , Please Restart Server ~")

        @self.socketio.on("client_check_resource_complete")
        def client_check_resource_complete_handle(data):
            data = self.rsaDecrypt(data)
            if data["now_global_epoch"] == self.global_model.now_global_epoch:
                emit("c_check_resource_complete", {"sid": request.sid}, broadcast=True, namespace="/ui")  # for ui
                time.sleep(sleep_time)
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
                        self.global_train_next_epoch(runnable_client_sids)
                    else:
                        self.wait_time += 1 if self.wait_time < 10 else 0
                        time.sleep(self.wait_time)
                        self.clients_check_resource()

        @self.socketio.on("halfway_client_check_resource_complete")
        def halfway_client_check_resource_complete_handle(data):
            data = self.rsaDecrypt(data)
            if data["now_global_epoch"] == self.global_model.now_global_epoch:
                sid = request.sid
                emit("c_check_resource_complete", {"sid": sid}, broadcast=True, namespace="/ui")  # for ui
                time.sleep(sleep_time)
                self.client_resource[sid] = data["loadavg"]
                loadavg = self.client_resource[sid]
                self.logger.debug("Client-sid:[{}] , Loadavg : {}".format(sid, loadavg))
                if float(loadavg) < self.CLIENT_SINGLE_MAX_LOADAVG:
                    self.logger.info("Client-sid:[{}] Runnable".format(sid))
                    self.wait_time = min(self.wait_time, 3)
                    self.halfway_train(sid)
                else:
                    self.logger.warning("Client-sid:[{}] Over-loadavg".format(sid))
                    self.wait_time += 1 if self.wait_time < 10 else 0
                    time.sleep(self.wait_time)
                    self.halfway_client_check_resource(sid)

        @self.socketio.on("client_update_complete")
        def client_update_complete_handle(data):
            data = self.rsaDecrypt(data)
            sid = request.sid
            self.logger.debug("Received Client-sid:[{}] Update-Data:{} ".format(sid, data))
            emit("c_train_complete", {"sid": request.sid, "gep": self.global_model.now_global_epoch}, broadcast=True,
                 namespace="/ui")  # for ui

            if self.global_model.now_global_epoch == data["now_global_epoch"]:
                # data["now_weights"] = copy.deepcopy(utils.pickle2obj(data["now_weights"]))
                self.client_update_datas[sid] = data
                # all clients upload complete
                if len(self.client_update_datas.keys()) == len(self.running_client_sids):
                    if len(self.running_client_sids) < self.NUM_CLIENTS <= len(self.ready_client_sids):
                        self.clients_check_resource()
                    if len(self.running_client_sids) == self.NUM_CLIENTS <= len(self.ready_client_sids):
                        emit("s_train_aggre", {"gep": self.global_model.now_global_epoch}, broadcast=True, namespace="/ui")  # for ui
                        receive_data_keys = list(list(self.client_update_datas.values())[0].keys())

                        self.global_model.update_global_weights(
                            [copy.deepcopy(utils.pickle2obj(client_data["now_weights"])) for client_data in self.client_update_datas.values()],
                            [client_data[constants.TRAIN_CONTRIB] for client_data in self.client_update_datas.values()])

                        global_train_loss, _ = self.global_model.get_aggre_loss_acc(
                            constants.TRAIN,
                            [client_data[constants.TRAIN_LOSS] for client_data in self.client_update_datas.values()],
                            None,
                            [client_data[constants.TRAIN_CONTRIB] for client_data in self.client_update_datas.values()],
                            False)

                        self.logger.info(
                            "Train -- GlobalEpoch:{} -- AvgLoss:{:.4f}".format(self.global_model.now_global_epoch, global_train_loss))
                        self.tbX.add_scalar("train/global_loss", global_train_loss, self.global_model.now_global_epoch)

                        if constants.TRAIN in receive_data_keys:
                            self._local_eval(constants.TRAIN)

                        if constants.VALIDATION in receive_data_keys:
                            self._local_eval(constants.VALIDATION)

                        if constants.TEST in receive_data_keys:
                            self._local_eval(constants.TEST)

                        now_weights_pickle = utils.obj2pickle(self.global_model.global_weights, self.global_model.weights_path)  # weights path
                        emit_data = {"now_global_epoch": self.global_model.now_global_epoch,
                                     "now_weights": now_weights_pickle}
                        if constants.TRAIN in self.GLOBAL_EVAL:
                            emit_data[constants.TRAIN] = self.global_model.now_global_epoch % self.GLOBAL_EVAL[constants.TRAIN][
                                constants.NUM] == 0
                        if constants.VALIDATION in self.GLOBAL_EVAL:
                            emit_data[constants.VALIDATION] = self.global_model.now_global_epoch % self.GLOBAL_EVAL[constants.VALIDATION][
                                constants.NUM] == 0
                        if constants.TEST in self.GLOBAL_EVAL:
                            emit_data[constants.TEST] = self.global_model.now_global_epoch % self.GLOBAL_EVAL[constants.TEST][
                                constants.NUM] == 0

                        emit("s_train_aggre_complete", {"gep": self.global_model.now_global_epoch}, broadcast=True, namespace="/ui")  # for ui
                        time.sleep(sleep_time)
                        self.client_eval_datas = dict()  # empty eval datas for next eval epoch
                        for sid in self.running_client_sids:
                            emit_data["sid"] = sid
                            emit("c_eval", {"sid": sid, "gep": self.global_model.now_global_epoch}, broadcast=True, namespace="/ui")  # for ui
                            time.sleep(sleep_time)
                            emit("eval_with_global_weights", self.rsaEncrypt(sid, emit_data), room=sid)
                            self.logger.info("Server Send Federated Weights To Client-sid:[{}]".format(sid))

        @self.socketio.on("train_process")
        def train_process_handle(data):
            # self.logger.debug("Received Client-sid:[{}] Train-process-data:{} ".format(request.sid, data))
            emit("ui_train_process", {"sid": request.sid, "gep": self.global_model.now_global_epoch, "process": data["process"]},
                 broadcast=True, namespace="/ui")  # for ui

        @self.socketio.on("eval_with_global_weights_complete")
        def eval_with_global_weights_complete_handle(data):
            data = self.rsaDecrypt(data)
            sid = request.sid
            self.logger.debug("Receive Client-sid:[{}] Eval Datas:{}".format(sid, data))
            emit("c_eval_complete", {"sid": request.sid, "gep": self.global_model.now_global_epoch}, broadcast=True, namespace="/ui")  # for ui
            time.sleep(sleep_time)

            self.client_eval_datas[sid] = data
            if len(self.client_eval_datas.keys()) == len(self.running_client_sids):
                if len(self.running_client_sids) < self.NUM_CLIENTS <= len(self.ready_client_sids):
                    self.clients_check_resource()
                if len(self.running_client_sids) == self.NUM_CLIENTS <= len(self.ready_client_sids):
                    emit("s_eval_aggre", {"gep": self.global_model.now_global_epoch}, broadcast=True, namespace="/ui")  # for ui
                    global_eval_types = list(list(self.client_eval_datas.values())[0].keys())

                    if constants.TRAIN in global_eval_types:
                        self._global_eval(constants.TRAIN)
                        self.global_model.update_best(constants.TRAIN)
                        tolerate_res = self.global_model.update_tolerate(constants.TRAIN)
                        if isinstance(tolerate_res, bool):
                            self.fin = tolerate_res

                    if constants.VALIDATION in global_eval_types:
                        self._global_eval(constants.VALIDATION)
                        self.global_model.update_best(constants.VALIDATION)
                        tolerate_res = self.global_model.update_tolerate(constants.VALIDATION)
                        if isinstance(tolerate_res, bool):
                            self.fin = tolerate_res

                    if constants.TEST in global_eval_types:
                        self._global_eval(constants.TEST)
                        self.global_model.update_best(constants.TEST)
                        tolerate_res = self.global_model.update_tolerate(constants.TEST)
                        if isinstance(tolerate_res, bool):
                            self.fin = tolerate_res

                    self.global_model.save_ckpt(self.SAVE_CKPT_EPOCH)

                    emit("s_eval_aggre_complete", {"gep": self.global_model.now_global_epoch}, broadcast=True, namespace="/ui")  # for ui
                    time.sleep(sleep_time)

                    if self.global_model.now_global_epoch >= self.NUM_GLOBAL_EPOCH > 0:
                        self.fin = True
                        self.logger.info("Go to NUM_GLOBAL_EPOCH:{}".format(self.NUM_GLOBAL_EPOCH))

                    if not self.fin:
                        # next global epoch
                        self.logger.info("Start Next Global-Epoch Training ...")
                    else:
                        emit("s_summary", broadcast=True, namespace="/ui")  # for ui
                        time.sleep(sleep_time)
                        self.global_model.fin_summary(global_eval_types)
                        emit("s_summary_complete", broadcast=True, namespace="/ui")  # for ui
                        time.sleep(sleep_time)

                    self.clients_check_resource()

        @self.socketio.on("eval_process")
        def eval_process_handle(data):
            # self.logger.debug("Received Client-sid:[{}] Eval-process-data:{} ".format(request.sid, data))
            emit("ui_eval_process",
                 {"sid": request.sid, "gep": self.global_model.now_global_epoch, "process": data["process"], "type": data["type"]},
                 broadcast=True, namespace="/ui")  # for ui

        @self.socketio.on("client_fin")
        def handle_client_fin(data):
            data = self.rsaDecrypt(data)
            sid = data["sid"]
            self.logger.info("Federated Learning Client-sid:[{}] Fin.".format(sid))
            disconnect(sid)
            if sid in self.ready_client_sids:
                self.ready_client_sids.remove(sid)
                emit("c_fin", {"sid": sid}, broadcast=True, namespace="/ui")  # for ui
            if sid in self.running_client_sids:
                self.running_client_sids.remove(sid)
            if sid in self.client_update_datas.keys():
                self.client_update_datas.pop(sid)
            if sid in self.client_eval_datas.keys():
                self.client_eval_datas.pop(sid)
            if len(self.ready_client_sids) == 0:
                self.tbX.close()
                self.logger.info("All Clients Fin. Federated Learning Server Fin.")
                emit("s_fin", broadcast=True, namespace="/ui")  # for ui
                time.sleep(5)
                self.socketio.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_config_path", type=str, required=True, help="path of server config")
    parser.add_argument("--host", type=str, help="optional server host , 'configs/base_config.yaml' has inited host")
    parser.add_argument("--port", type=int, help="optional server port , 'configs/base_config.yaml' has inited port")

    args = parser.parse_args()

    assert osp.exists(args.server_config_path), "{} not exist".format(args.server_config_path)

    try:
        config = utils.load_json(args.server_config_path)
        config[constants.PID] = os.getpid()
        with open(args.server_config_path, "w+") as f:
            json.dump(config, f, indent=4)
        server = FederatedServer(args.server_config_path, args.host, args.port)
        server.start()
    except ConnectionError:
        print("server connect error")
