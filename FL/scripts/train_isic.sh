#!/bin/bash
gpu=5
num_users=20
cd ..
#for num_users in 20 50 200; do
#  python3 main.py --dataset mnist --model mlp --epochs 1000 --lr 0.01 --eval_interval 10 --num_users ${num_users} --client_frac 0.1 --local_ep 2 --local_bs 50 --iid --gpu ${gpu}
#done

#isic  rmsprop
#python3 main.py --dataset isic --num_channels 3 --num_classes 2 --model unet --epochs 100 --lr 1e-3 --weight_decay 1e-8 --momentum 0.9 --eval_interval 5 --train_bs 8 --test_bs 16 --gpu ${gpu} --verbose

#isic adam
#python3 main.py --dataset isic --num_channels 3 --num_classes 2 --model unet --epochs 200 --lr 1e-3 --train_bs 8 --test_bs 16 --eval_interval 5 --gpu ${gpu} --verbose

#non-Fed
python3 main.py \
  --dataset isic \
  --num_channels 3 \
  --num_classes 1 \
  --model unet \
  --epochs 100 \
  --train_bs 20 \
  --test_bs 40 \
  --eval_interval 5 \
  --gpu ${gpu} \
  --verbose

#i.i.d
python3 main.py \
  --dataset isic \
  --num_channels 3 \
  --num_classes 1 \
  --model unet \
  --epochs 100 \
  --iid \
  --num_users ${num_users} \
  --client_frac 1 \
  --all_clients \
  --local_ep 2 \
  --local_bs 16 \
  --train_bs 32 \
  --test_bs 32 \
  --eval_interval 5 \
  --gpu ${gpu} \
  --verbose
