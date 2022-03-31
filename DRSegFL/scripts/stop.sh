#!/bin/bash

cd ..

read -p "input config_path : " config_path
python3 stop.py --config_path "${config_path}"
