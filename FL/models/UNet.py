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
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=0, stride=1, dilation=1, batch_norm=True, dropout=False):
        super(encode_block, self).__init__()
        self.encoding_block = nn.Sequential(
            *filter(None, [
                nn.ReflectionPad2d((kernel_size - 1) // 2),
                nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.PReLU(),
                nn.BatchNorm2d(out_dim) if batch_norm else None,
                nn.ReflectionPad2d((kernel_size - 1) // 2),
                nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.PReLU(),
                nn.BatchNorm2d(out_dim) if batch_norm else None,
                nn.Dropout() if dropout else None,
            ])
        )

    def forward(self, input):
        return self.encoding_block(input)


class decode_block(nn.Module):
    def __init__(self, in_dim, out_dim, batch_norm=False, upsampling=True):
        super(decode_block, self).__init__()
        # 采样策略
        if upsampling:
            # 使用插值法改变图像size
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_dim, out_dim, kernel_size=1)
            )
        else:
            self.up = nn.Sequential(
                # output=(input - 1) * stride + kernel_size - 2 * padding + output_padding
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2, padding=0, output_padding=0)
            )
        self.conv = encode_block(in_dim, out_dim, batch_norm=batch_norm)

    def forward(self, crop_input, upsampling_input):
        upsampling_output = self.up(upsampling_input)
        crop_output = F.interpolate(crop_input, size=upsampling_output.shape[2:], mode="bilinear")
        return self.conv(torch.cat([crop_output, upsampling_output], 1))


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        # contract path
        self.encoder_block1 = encode_block(3, 64)
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
        self.final = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Sigmoid(),
        )

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


if __name__ == "__main__":
    unet = UNet(num_classes=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet.to(device)
    summary(unet, (3, 256, 256))
