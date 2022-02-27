#!/bin/bash

cd ..

read -p "input now client_config_path : " client_config_path

nohup python3 fl_client.py --client_config_path "${client_config_path}" >> all.log 2>&1 &
