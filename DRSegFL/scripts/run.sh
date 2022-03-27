#!/bin/bash
now_time=$(date '+%Y-%m-%d-%H-%M-%S')
seconds_left=8

cd ..

read -p "num_clients : " num_clients
read -p "skip_generate_configs : " skip_generate_configs
if [ ${skip_generate_configs} = "N" -o ${skip_generate_configs} = "n" ]; then
  cd configs
  python3 config_generate.py --num_clients ${num_clients}
  cd ..
fi
read -p "input server_config_path : " server_config_path
nohup python3 fl_server.py --server_config_path "${server_config_path}" >>"${now_time}.log" 2>&1 &

echo "please wait for a while ..."
while [ $seconds_left -gt 0 ]; do
  sleep 1
  seconds_left=$(($seconds_left - 1))
done

for i in $(seq 1 ${num_clients}); do
  read -p "input client_${i}_config_path : " client_config_path
  nohup python3 fl_client.py --client_config_path "${client_config_path}" >>"${now_time}.log" 2>&1 &
done
