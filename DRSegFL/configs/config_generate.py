#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import argparse
import copy
import json
import os
import os.path as osp
import sys

root_dir_name = osp.dirname(osp.dirname(sys.path[0]))  # ...Neko-ML/
sub_root_dir_name = osp.dirname(sys.path[0])  # ...DRSegFL/
now_dir_name = sys.path[0]  # ...configs/

sys.path.append(root_dir_name)

from DRSegFL import utils, constants, datasets
from DRSegFL.logger import Logger


def generate_dataset_txt(config, dataset_dir, dataset_txt_path, dataset_type):
    """
    Todo: add dataset , modify below
    :param config:
    :param dataset_dir:
    :param dataset_txt_path:
    :param dataset_type:
    """
    is_augment = config[constants.MODEL]["data_augment"]
    # ISIC
    if config[constants.MODEL][constants.NAME_DATASET] == constants.ISIC:
        img_dir = osp.join(dataset_dir, "image")
        target_dir = osp.join(dataset_dir, "mask")

        if is_augment and dataset_type == constants.TRAIN:
            datasets.dataset_augment(img_dir, target_dir, "jpg", "png", dataset_type)

        if isinstance(dataset_txt_path, list):
            datasets.iid_dataset_txt_generate(img_dir, "jpg", target_dir, "png", dataset_txt_path, is_augment)
        else:
            datasets.dataset_txt_generate(img_dir, "jpg", target_dir, "png", dataset_txt_path, is_augment)
    # DDR
    elif config[constants.MODEL][constants.NAME_DATASET] == constants.DDR:
        no_bg_classes = copy.deepcopy(config[constants.MODEL][constants.CLASSES])
        if "bg" in no_bg_classes:  # remove background(bg)
            no_bg_classes.remove("bg")

        img_dir = osp.join(dataset_dir, "image")
        target_dir = osp.join(dataset_dir, "label")
        ann_dir = osp.join(dataset_dir, "annotation")

        datasets.labels2annotations(img_dir, target_dir, ann_dir, "jpg", "tif", no_bg_classes, dataset_type, force=False)

        if is_augment and dataset_type == constants.TRAIN:
            datasets.dataset_augment(img_dir, ann_dir, "jpg", "png", dataset_type)

        if isinstance(dataset_txt_path, list):
            datasets.iid_dataset_txt_generate(img_dir, "jpg", ann_dir, "png", dataset_txt_path, is_augment)
        else:
            datasets.dataset_txt_generate(img_dir, "jpg", ann_dir, "png", dataset_txt_path, is_augment)
    # DDR SingleLesion
    elif config[constants.MODEL][constants.NAME_DATASET] in [constants.DDR_EX, constants.DDR_HE, constants.DDR_MA, constants.DDR_SE]:
        img_dir = osp.join(dataset_dir, "image")
        target_dir = osp.join(dataset_dir, "label")

        if is_augment and dataset_type == constants.TRAIN:
            datasets.dataset_augment(img_dir, target_dir, "jpg", "png", dataset_type)

        if isinstance(dataset_txt_path, list):
            datasets.iid_dataset_txt_generate(img_dir, "jpg", target_dir, "png", dataset_txt_path, is_augment)
        else:
            datasets.dataset_txt_generate(img_dir, "jpg", target_dir, "png", dataset_txt_path, is_augment)
    else:
        raise ValueError(config[constants.MODEL][constants.NAME_DATASET])


def generate(base_config_path="./base_config.yaml", num_clients=None, logger=None):
    """
    generate configs
    :param base_config_path:
    :param num_clients:
    :param logger: default is None ,which will use print
    :return:
    """
    logger = Logger(logger)
    if not osp.exists(base_config_path):
        logger.error("please make sure base_config.yaml in {}".format(base_config_path))
        exit(-1)

    config = utils.load_yaml(base_config_path)
    config[constants.SERVER][constants.NUM_CLIENTS] = config[constants.SERVER][constants.NUM_CLIENTS] \
        if num_clients is None else num_clients
    logger.info("Generating configs ...")

    now_time = utils.get_now_time()
    now_day = utils.get_now_day()

    config_dir = osp.join(sub_root_dir_name, "generates", "configs",
                          config[constants.MODEL][constants.NAME_MODEL],
                          config[constants.MODEL][constants.NAME_DATASET], now_day, now_time)
    logfile_dir = osp.join(sub_root_dir_name, "logs",
                           config[constants.MODEL][constants.NAME_MODEL],
                           config[constants.MODEL][constants.NAME_DATASET], now_day, now_time)
    weights_dir = osp.join(sub_root_dir_name, "saves", "weights",
                           config[constants.MODEL][constants.NAME_MODEL],
                           config[constants.MODEL][constants.NAME_DATASET], now_day, now_time)
    best_weights_dir = osp.join(weights_dir, "best")
    predict_dir = osp.join(sub_root_dir_name, "saves", "predict", config[constants.MODEL][constants.NAME_MODEL],
                           config[constants.MODEL][constants.NAME_DATASET])
    generate_dataset_txt_dir = osp.join(sub_root_dir_name, "generates", "datasets",
                                        config[constants.MODEL][constants.NAME_DATASET], now_day, now_time)

    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(logfile_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(best_weights_dir, exist_ok=True)
    os.makedirs(generate_dataset_txt_dir, exist_ok=True)

    train_dataset_dir = config[constants.MODEL][constants.DIR_DATASET][constants.TRAIN]
    val_dataset_dir = config[constants.MODEL][constants.DIR_DATASET][constants.VALIDATION] \
        if constants.VALIDATION in config[constants.MODEL][constants.DIR_DATASET] else None
    test_dataset_dir = config[constants.MODEL][constants.DIR_DATASET][constants.TEST] \
        if constants.TEST in config[constants.MODEL][constants.DIR_DATASET] else None

    if not osp.exists(train_dataset_dir):
        logger.error("please put train dataset in {}".format(train_dataset_dir))
        exit(-1)
    if val_dataset_dir and not osp.exists(val_dataset_dir):
        logger.error("please put val dataset in {}".format(val_dataset_dir))
        exit(-1)
    if test_dataset_dir and not osp.exists(test_dataset_dir):
        logger.error("please put test dataset in {}".format(test_dataset_dir))
        exit(-1)

    server_config_path = osp.join(config_dir, "server_config.json")
    client_configs_path = [osp.join(config_dir, "client_{}_config.json".format(i)) for i in
                           range(1, config[constants.SERVER][constants.NUM_CLIENTS] + 1)]

    iid_train_dataset_txt_path = [osp.join(generate_dataset_txt_dir, "client_{}_train.txt".format(i)) for i in
                                  range(1, config[constants.SERVER][constants.NUM_CLIENTS] + 1)]
    val_dataset_txt_path = osp.join(generate_dataset_txt_dir,
                                    "{}.txt".format(constants.VALIDATION)) if val_dataset_dir else None
    test_dataset_txt_path = osp.join(generate_dataset_txt_dir,
                                     "{}.txt".format(constants.TEST)) if test_dataset_dir else None

    generate_dataset_txt(config, train_dataset_dir, iid_train_dataset_txt_path, dataset_type=constants.TRAIN)
    if val_dataset_txt_path:
        generate_dataset_txt(config, val_dataset_dir, val_dataset_txt_path, dataset_type=constants.VALIDATION)
    if test_dataset_txt_path:
        generate_dataset_txt(config, test_dataset_dir, test_dataset_txt_path, dataset_type=constants.TEST)

    # partial model config -> server/client config
    config[constants.SERVER].update(config[constants.MODEL])
    del config[constants.SERVER][constants.DIR_DATASET]

    config[constants.CLIENT].update(config[constants.MODEL])
    del config[constants.CLIENT][constants.DIR_DATASET]
    # partial server config -> client config
    config[constants.CLIENT][constants.HOST] = config[constants.SERVER][constants.HOST]
    config[constants.CLIENT][constants.PORT] = config[constants.SERVER][constants.PORT]

    with open(server_config_path, "w+") as f:
        config[constants.SERVER][constants.PATH_LOGFILE] = osp.join(logfile_dir, "server.log")
        config[constants.SERVER][constants.PATH_WEIGHTS] = osp.join(weights_dir, "fed_c{}_ep{}.pkl".format(
            config[constants.SERVER][constants.NUM_CLIENTS], config[constants.SERVER][constants.EPOCH]))
        config[constants.SERVER][constants.DIR_PREDICT] = predict_dir
        config[constants.SERVER][constants.PATH_BEST_WEIGHTS] = {
            constants.TRAIN: osp.join(best_weights_dir, "fed_train_c{}_ep{}.pt".format(
                config[constants.SERVER][constants.NUM_CLIENTS], config[constants.SERVER][constants.EPOCH])),
            constants.VALIDATION: osp.join(best_weights_dir, "fed_val_c{}_ep{}.pt".format(
                config[constants.SERVER][constants.NUM_CLIENTS], config[constants.SERVER][constants.EPOCH])),
            constants.TEST: osp.join(best_weights_dir, "fed_test_c{}_ep{}.pt".format(
                config[constants.SERVER][constants.NUM_CLIENTS], config[constants.SERVER][constants.EPOCH]))}
        json.dump(config[constants.SERVER], f, indent=4)

    for i in range(config[constants.SERVER][constants.NUM_CLIENTS]):
        with open(client_configs_path[i], "w+") as f:
            config[constants.CLIENT][constants.PATH_LOGFILE] = osp.join(logfile_dir,
                                                                        "client_{}.log".format(i + 1))
            config[constants.CLIENT][constants.PATH_WEIGHTS] = osp.join(weights_dir,
                                                                        "local_c{}.pkl".format(i + 1))
            config[constants.CLIENT][constants.TRAIN] = iid_train_dataset_txt_path[i]
            config[constants.CLIENT][constants.VALIDATION] = val_dataset_txt_path
            config[constants.CLIENT][constants.TEST] = test_dataset_txt_path
            json.dump(config[constants.CLIENT], f, indent=4)

    logger.info("Generate completed ~")
    logger.info("server_config_path:{}".format(server_config_path))
    logger.info("num_client_configs:{}".format(len(client_configs_path)))
    [logger.info("client_config_{}_path:{}".format(i + 1, client_configs_path[i])) for i in range(len(client_configs_path))]
    return server_config_path, client_configs_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config_path", type=str, default="./base_config.yaml",
                        help="optional base_config_path,default is ./base_config.yaml")
    parser.add_argument("--num_clients", type=int,
                        help="optional clients_num,'configs/base_config.yaml' has inited clients_num")
    args = parser.parse_args()

    generate(base_config_path=args.base_config_path, num_clients=args.num_clients)
