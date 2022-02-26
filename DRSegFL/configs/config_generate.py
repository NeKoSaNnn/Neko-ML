#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import json
import os
import os.path as osp
import sys

root_dir_name = osp.dirname(osp.dirname(sys.path[0]))  # ...Neko-ML/
sub_root_dir_name = osp.dirname(sys.path[0])  # ...DRSegFL/
now_dir_name = sys.path[0]  # ...configs/

sys.path.append(root_dir_name)

from DRSegFL import utils, constants, datasets


def generate():
    if not osp.exists("./base_config.yaml"):
        print("please make sure base_config.yaml in now dir")
        exit(-1)

    config = utils.load_yaml("./base_config.yaml")

    now_time = utils.get_now_time()
    now_day = utils.get_now_day()

    config_dir = osp.join(sub_root_dir_name, "generates", "configs",
                          config[constants.MODEL][constants.NAME_MODEL],
                          config[constants.MODEL][constants.NAME_DATASET], now_day)
    logfile_dir = osp.join(sub_root_dir_name, "logs",
                           config[constants.MODEL][constants.NAME_MODEL],
                           config[constants.MODEL][constants.NAME_DATASET], now_day)
    weights_dir = osp.join(sub_root_dir_name, "saves", "weights",
                           config[constants.MODEL][constants.NAME_MODEL],
                           config[constants.MODEL][constants.NAME_DATASET], now_day)

    train_dataset_dir = config[constants.MODEL][constants.DIR_DATASET][constants.TRAIN]
    val_dataset_dir = config[constants.MODEL][constants.DIR_DATASET][constants.VALIDATION] \
        if constants.VALIDATION in config[constants.MODEL][constants.DIR_DATASET] else None
    test_dataset_dir = config[constants.MODEL][constants.DIR_DATASET][constants.TEST] \
        if constants.TEST in config[constants.MODEL][constants.DIR_DATASET] else None

    generate_dataset_txt_dir = osp.join(sub_root_dir_name, "generates", "datasets",
                                        config[constants.MODEL][constants.NAME_DATASET], now_day)

    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(logfile_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(generate_dataset_txt_dir, exist_ok=True)

    if not osp.exists(train_dataset_dir):
        print("please put train dataset in {}".format(train_dataset_dir))
        exit(-1)
    if val_dataset_dir is not None and not osp.exists(val_dataset_dir):
        print("please put val dataset in {}".format(val_dataset_dir))
        exit(-1)
    if test_dataset_dir is not None and not osp.exists(test_dataset_dir):
        print("please put test dataset in {}".format(test_dataset_dir))
        exit(-1)

    server_config_path = osp.join(config_dir, "server_config_{}.json".format(now_time))
    client_configs_path = [osp.join(config_dir, "client_{}_config_{}.json".format(i, now_time)) for i in
                           range(config[constants.SERVER][constants.NUM_CLIENTS])]

    iid_train_dataset_txt_path = [osp.join(generate_dataset_txt_dir, "client_{}_train.txt".format(i)) for i in
                                  range(config[constants.SERVER][constants.NUM_CLIENTS])]
    val_dataset_txt_path = osp.join(generate_dataset_txt_dir,
                                    "{}.txt".format(constants.VALIDATION)) if val_dataset_dir is not None else None
    test_dataset_txt_path = osp.join(generate_dataset_txt_dir,
                                     "{}.txt".format(constants.TEST)) if test_dataset_dir is not None else None

    # Todo: change dataset , modify below
    datasets.iid_dataset_txt_generate(osp.join(train_dataset_dir, "image"), "jpg", osp.join(train_dataset_dir, "mask"),
                                      "png", iid_train_dataset_txt_path)
    if val_dataset_txt_path is not None:
        datasets.dataset_txt_generate(osp.join(val_dataset_dir, "image"), "jpg", osp.join(val_dataset_dir, "mask"),
                                      "png", val_dataset_txt_path)
    if test_dataset_txt_path is not None:
        datasets.dataset_txt_generate(osp.join(test_dataset_dir, "image"), "jpg", osp.join(test_dataset_dir, "mask"),
                                      "png", test_dataset_txt_path)

    # partial model config -> server/client config
    config[constants.SERVER].update(config[constants.MODEL])
    del config[constants.SERVER][constants.DIR_DATASET]

    config[constants.CLIENT].update(config[constants.MODEL])
    del config[constants.CLIENT][constants.DIR_DATASET]
    # partial server config -> client config
    config[constants.CLIENT][constants.HOST] = config[constants.SERVER][constants.HOST]
    config[constants.CLIENT][constants.PORT] = config[constants.SERVER][constants.PORT]

    with open(server_config_path, "w+") as f:
        config[constants.SERVER][constants.PATH_LOGFILE] = osp.join(logfile_dir, "fed_server.log")
        config[constants.SERVER][constants.PATH_WEIGHTS] = osp.join(weights_dir,
                                                                    "fed_c{}_ep{}_{}.pkl".format(
                                                                        config[constants.SERVER][constants.NUM_CLIENTS],
                                                                        config[constants.SERVER][constants.EPOCH],
                                                                        now_time))
        json.dump(config[constants.SERVER], f, indent=4)

    for i in range(config[constants.SERVER][constants.NUM_CLIENTS]):
        with open(client_configs_path[i], "w+") as f:
            config[constants.CLIENT][constants.PATH_LOGFILE] = osp.join(logfile_dir, "fed_client_{}.log".format(i))
            # Todo: change dataset , modify below
            config[constants.CLIENT][constants.TRAIN] = iid_train_dataset_txt_path[i]
            config[constants.CLIENT][constants.VALIDATION] = val_dataset_txt_path
            config[constants.CLIENT][constants.TEST] = test_dataset_txt_path
            json.dump(config[constants.CLIENT], f, indent=4)

    print("Generate complete ...")
    print("server_config_path:{}".format(server_config_path))
    print("num_client_configs:{}".format(len(client_configs_path)))
    [print("client_config_{}_path:{}".format(i, client_configs_path[i])) for i in range(len(client_configs_path))]
    return server_config_path, client_configs_path


if __name__ == "__main__":
    generate()
