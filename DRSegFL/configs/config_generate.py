#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import json
import os
import os.path as osp

import yaml

from DRSegFL import utils

with open("base_config.yaml", encoding="UTF-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config_dir = osp.join(osp.dirname(osp.dirname(__file__)), "saves", "configs", config["model"]["model_name"],
                      config["model"]["dataset_name"], utils.get_now_day())
logfile_dir = osp.join(osp.dirname(osp.dirname(__file__)), "logs", config["model"]["model_name"],
                       config["model"]["dataset_name"], utils.get_now_day())
weights_dir = osp.join(osp.dirname(osp.dirname(__file__)), "saves", "weights", config["model"]["model_name"],
                       config["model"]["dataset_name"], utils.get_now_day())

os.makedirs(config_dir, exist_ok=True)
os.makedirs(logfile_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

now_time = utils.get_now_time()
server_config_path = osp.join(config_dir, "server_config_{}.json".format(now_time))
client_configs_path = [osp.join(config_dir, "client_{}_config_{}.json".format(i, now_time)) for i in
                       range(config["server"]["num_clients"])]

config["server"].update(config["model"])
config["client"].update(config["model"])

with open(server_config_path, "w+") as f:
    config["server"]["logfile_path"] = osp.join(logfile_dir, "fed_server.log")
    config["server"]["weights_path"] = osp.join(weights_dir,
                                                "fed_c{}_ep{}_{}.pkl".format(config["server"]["num_clients"],
                                                                             config["server"]["epoch"], now_time))
    json.dump(config["server"], f, indent=4)

for i in range(config["server"]["num_clients"]):
    with open(client_configs_path[i], "w+") as f:
        config["client"]["logfile_path"] = osp.join(logfile_dir, "fed_client_{}.log".format(i))
        config["client"]["train"] = "..."
        config["client"]["val"] = "..."
        config["client"]["test"] = "..."
        json.dump(config["client"], f, indent=4)
