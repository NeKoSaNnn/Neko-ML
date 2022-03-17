#!/bin/bash

cd ..

read -p "logdir : " logdir

tensorboard --logdir="${logdir}" --port=6006
