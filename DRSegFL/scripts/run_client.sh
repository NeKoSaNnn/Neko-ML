#!/bin/bash
set -x
set -e

cd ..

read -p "input now client_config_path : " client_config_path

python3 fl_client.py --client_config_path "${client_config_path}"
