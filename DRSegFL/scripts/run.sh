#!/bin/bash

cd ..

read -p "num_clients : " num_clients
cd configs
python3 config_generate.py --num_clients ${num_clients}
cd ..

read -p "input server_config_path : " server_config_path
nohup python3 fl_server.py --server_config_path "${server_config_path}" >all.log 2>&1 &

for i in $(seq 1 ${num_clients}); do
  read -p "input now client_config_path : " client_config_path
  nohup python3 fl_client.py --client_config_path "${client_config_path}" >all.log 2>&1 &
done
