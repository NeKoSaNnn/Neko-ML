#!/bin/bash
now_time=$(date '+%Y-%m-%d-%H')

cd ..

read -p "input server_config_path : " server_config_path

nohup python3 fl_server.py --server_config_path "${server_config_path}" >>"${now_time}.log" 2>&1 &
