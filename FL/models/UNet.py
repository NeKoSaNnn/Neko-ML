#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class encode_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, dilation=1, batch_norm=True,
                 dropout=False):
        super(encode_block, self).__init__()
        self.encoding_block = nn.Sequential(
            *filter(None, [
                nn.ReflectionPad2d((kernel_size - 1) // 2),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                          dilation=dilation),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels) if batch_norm else None,
                nn.ReflectionPad2d((kernel_size - 1) // 2),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                          dilation=dilation),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels) if batch_norm else None,
                nn.Dropout() if dropout else None,
            ])
        )

    def forward(self, input):
        return self.encoding_block(input)


class decode_block(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False, upsampling=True):
        super(decode_block, self).__init__()
        # 采样策略
        if upsampling:
            # 使用插值法改变图像size
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        else:
            self.up = nn.Sequential(
                # output=(input - 1) * stride + kernel_size - 2 * padding + output_padding
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0)
            )
        self.conv = encode_block(in_channels, out_channels, batch_norm=batch_norm)

    def forward(self, crop_input, upsampling_input):
        upsampling_output = self.up(upsampling_input)
        crop_output = F.interpolate(crop_input, size=upsampling_output.shape[2:], mode="bilinear")
        return self.conv(torch.cat([crop_output, upsampling_output], 1))


class UNet1(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(UNet, self).__init__()

        self.in_channels = num_channels
        self.num_classes = num_classes

        # contract path
        self.encoder_block1 = encode_block(num_channels, 64)
        self.maxpool1 = nn.MaxPool2d(2)

        self.encoder_block2 = encode_block(64, 128)
        self.maxpool2 = nn.MaxPool2d(2)

        self.encoder_block3 = encode_block(128, 256)
        self.maxpool3 = nn.MaxPool2d(2)

        self.encoder_block4 = encode_block(256, 512)
        self.maxpool4 = nn.MaxPool2d(2)

        # center
        self.center = encode_block(512, 1024)

        # expand path
        self.decoder_block4 = decode_block(1024, 512)
        self.decoder_block3 = decode_block(512, 256)
        self.decoder_block2 = decode_block(256, 128)
        self.decoder_block1 = decode_block(128, 64)

        # final
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, input):
        # encoding
        conv1 = self.encoder_block1(input)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.encoder_block2(maxpool1)
        maxpool2 = self.maxpool1(conv2)
        conv3 = self.encoder_block3(maxpool2)
        maxpool3 = self.maxpool1(conv3)
        conv4 = self.encoder_block4(maxpool3)
        maxpool4 = self.maxpool1(conv4)

        # center
        center = self.center(maxpool4)

        # decoding
        decode4 = self.decoder_block4(conv4, center)
        decode3 = self.decoder_block3(conv3, decode4)
        decode2 = self.decoder_block2(conv2, decode3)
        decode1 = self.decoder_block1(conv1, decode2)

        final = F.interpolate(self.final(decode1), size=input.shape[2:], mode="bilinear")

        return final


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

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_channels, num_classes, upsampling=True):
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
