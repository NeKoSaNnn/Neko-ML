#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import torch.nn as nn


class MnistCNN(nn.Module):
    def __init__(self, args):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(args.num_channels, 10, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout2d(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(50, args.num_classes)
        )

    def forward(self, x):
        # x -> (n,1,28,28)
        x = self.conv1(x)  # x -> (n,10,24,24) -> (n,10,12,12)
        x = self.conv2(x)  # x -> (n,20,8,8) -> (n,20,4,4)
        x = x.flatten(1)  # x -> (n,320)
        x = self.fc1(x)  # x -> (n,50)
        x = self.fc2(x)  # x -> (n,10)
        return x


class CifarCNN(nn.Module):
    def __init__(self, args):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(args.num_channels, 6, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout2d(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Dropout2d(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        # x -> (n,3,32,32)
        x = self.conv1(x)  # x -> (n,6,28,28) -> (n,6,14,14)
        x = self.conv2(x)  # x -> (n,16,10,10) -> (n,16,5,5)
        x = x.flatten(1)  # x -> (n,400)
        x = self.fc1(x)  # x -> (n,120)
        x = self.fc2(x)  # x -> (n,84)
        x = self.fc3(x)  # x -> (n,10)

        return x
