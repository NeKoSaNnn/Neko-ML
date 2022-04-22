#!/bin/bash

cd ..

read -p "logdir : " logdir
read -p "port : " port

tensorboard --logdir="${logdir}" --port="${port}"
