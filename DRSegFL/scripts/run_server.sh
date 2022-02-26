#!/bin/bash

cd ..

read -p "input server_config_path : " server_config_path

python3 fl_server.py --server_config_path "${server_config_path}"
