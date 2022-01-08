#!/bin/bash

for NUM_USERS in 20 50 200; do
  #mnist
  python3 main.py --dataset mnist --model mlp --epochs 1000 --lr 0.01 --eval_interval 10 --num_users ${NUM_USERS} --client_frac 0.1 --local_ep 2 --local_bs 50 --iid

  python3 main.py --dataset mnist --model mlp --epochs 1000 --lr 0.01 --eval_interval 10 --num_channels 1

  #cifar
  python3 main.py --dataset cifar10 --model cnn --epochs 1000 --lr 0.01 --eval_interval 10 --num_users ${NUM_USERS} --client_frac 0.1 --local_ep 2 --local_bs 50 --iid

  python3 main.py --dataset cifar10 --model cnn --epochs 1000 --lr 0.01 --eval_interval 10
done

