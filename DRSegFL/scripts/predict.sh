#!/bin/bash
cd ..

python3 predict.py

#read -p "input predict config_path : " config_path
#read -p "input predict weights_path : " weights_path
#read -p "input predict predict_img_path : " predict_img_path
#read -p "input predict ground_truth_path : " ground_truth_path

#python3 predict.py \
#  --config_path "${config_path}" \
#  --weights_path "${weights_path}" \
#  --predict_img_path "${predict_img_path}" \
#  --ground_truth_path "${ground_truth_path}"
