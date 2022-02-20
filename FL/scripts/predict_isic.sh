#!/bin/bash
gpu=0
num_classes=1
cd ..

python3 predict.py \
  --dataset isic \
  --num_channels 3 \
  --num_classes ${num_classes} \
  --model unet \
  --gpu ${gpu}
