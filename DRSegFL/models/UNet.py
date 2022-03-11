#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.runner import load_checkpoint
from torchsummary import summary


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, middle_channels=None, batch_norm=True, dropout=False):
        super(DoubleConv, self).__init__()
        if not middle_channels:
            middle_channels = out_channels
        self.double_conv = nn.Sequential(*filter(None, [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels) if batch_norm else None,
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if batch_norm else None,
            nn.ReLU(inplace=True),
            nn.Dropout() if dropout else None]))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, upsampling=True):
        super(Up, self).__init__()
        if upsampling:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # (C,H,W)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [torch.div(diffX, 2, rounding_mode="floor"), diffX - torch.div(diffX, 2, rounding_mode="floor"),
                        torch.div(diffY, 2, rounding_mode="floor"), diffY - torch.div(diffX, 2, rounding_mode="floor")])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_channels: int, num_classes: int, upsampling=True, pretrain=None):
        super(UNet, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.upsampling = upsampling

        self.input_conv = DoubleConv(num_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, upsampling)
        self.up2 = Up(512, 128, upsampling)
        self.up3 = Up(256, 64, upsampling)
        self.up4 = Up(128, 64, upsampling)
        self.output_conv = OutConv(64, num_classes)

        self.init_weights(pretrain)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m.weight, 1)
                    constant_init(m.bias, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.output_conv(x)
        return x


if __name__ == "__main__":
    unet = UNet(num_channels=3, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet.to(device)
    summary(unet, (3, 256, 256))
