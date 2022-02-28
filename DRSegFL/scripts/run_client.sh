#!/bin/bash
now_time=$(date '+%Y-%m-%d-%H-%M-%S')

cd ..

read -p "input now client_config_path : " client_config_path

nohup python3 fl_client.py --client_config_path "${client_config_path}" >>"${now_time}.log" 2>&1 &
