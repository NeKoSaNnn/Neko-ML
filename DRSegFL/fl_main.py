#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""

import argparse
import os.path as osp
import sys

root_dir_name = osp.dirname(sys.path[0])  # ...Neko-ML/
now_dir_name = sys.path[0]  # ...DRSegFL/
sys.path.append(root_dir_name)

from DRSegFL import fl_client, fl_server
from DRSegFL.configs import config_generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, help="optional server host , 'configs/base_config.yaml' has inited host")
    parser.add_argument("--port", type=str, help="optional server port , 'configs/base_config.yaml' has inited port")
    args = parser.parse_args()

    server_config_path, client_configs_path = config_generate.generate()

    assert osp.exists(server_config_path), "{} not exist".format(server_config_path)

    server = fl_server.FederatedServer(server_config_path, args.host, args.port)
    server.start()
    for client_config_path in client_configs_path:
        assert osp.exists(client_config_path), "{} not exist".format(client_config_path)
        client = fl_client.FederatedClient(client_config_path, args.host, args.port)
