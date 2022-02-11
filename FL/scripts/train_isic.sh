#!/bin/bash
gpu=0
cd ..
#for NUM_USERS in 20 50 200; do
#  python3 main.py --dataset mnist --model mlp --epochs 1000 --lr 0.01 --eval_interval 10 --num_users ${NUM_USERS} --client_frac 0.1 --local_ep 2 --local_bs 50 --iid --gpu ${gpu}
#done

#isic  rmsProp
#python3 main.py --dataset isic --num_channels 3 --num_classes 2 --model unet --epochs 100 --lr 1e-3 --weight_decay 1e-8 --momentum 0.9 --eval_interval 5 --train_bs 8 --test_bs 16 --gpu ${gpu} --verbose

#isic Adam
#python3 main.py --dataset isic --num_channels 3 --num_classes 2 --model unet --epochs 200 --lr 1e-3 --train_bs 8 --test_bs 16 --eval_interval 5 --gpu ${gpu} --verbose

python3 main.py --dataset isic --num_channels 3 --num_classes 1 --model unet --epochs 1000 --train_bs 8 --test_bs 16 --eval_interval 5 --gpu ${gpu} --verbose
