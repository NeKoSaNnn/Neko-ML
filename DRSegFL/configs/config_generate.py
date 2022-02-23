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

config_path = osp.join(osp.dirname(__file__), config["model"]["model_name"], config["model"]["dataset_name"],
                       utils.get_now_day())
os.makedirs(config_path, exist_ok=True)

server_config = osp.join(config_path, "server_config.json")
client_configs = [osp.join(config_path, "client_{}_config.json".format(i)) for i in
                  range(1, config["server"]["num_clients"] + 1)]

config["server"].update(config["model"])
config["client"].update(config["model"])

with open(server_config, "w+") as f:
    json.dump(config["server"], f, indent=4)

for i in range(config["server"]["num_clients"]):
    with open(client_configs[i], "w+") as f:
        now_config = config["client"]
        now_config["log_filename"] = "fed_log_client_{}.log".format(i)
        json.dump(now_config, f, indent=4)
