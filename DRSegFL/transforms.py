#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Resize(object):
    def __init__(self, size: int, interpolation=F.InterpolationMode.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, target=None):
        if image.size[0] == image.size[1] == self.size:
            return image
        image = F.resize(image, [self.size, self.size], interpolation=self.interpolation)
        if target is not None:
            if target.size[0] == target.size[1] == self.size:
                return target
            target = F.resize(target, [self.size, self.size], interpolation=self.interpolation)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if target is not None:
                target = F.hflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image, target=None):
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        if target is not None:
            target = F.crop(target, *crop_params)
        return image, target


class RandomResizedCrop(object):
    def __init__(self, size: int, prob=1.0, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=F.InterpolationMode.BILINEAR):
        self.size = size
        self.prob = prob
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, image, target=None):
        if self.prob is None or (isinstance(self.prob, float) and random.random() < self.prob):
            rrc_params = T.RandomResizedCrop.get_params(image, self.scale, self.ratio)
            image = F.resized_crop(image, *rrc_params, size=[self.size, self.size], interpolation=self.interpolation)
            if target is not None:
                target = F.resized_crop(target, *rrc_params, size=[self.size, self.size], interpolation=self.interpolation)
        else:
            image, target = CenterCrop(min(image.size))(image, target)
            image, target = Resize(self.size, self.interpolation)(image, target)
        return image, target


class CenterCrop(object):
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image, target=None):
        image = F.center_crop(image, [self.size])
        if target is not None:
            target = F.center_crop(target, [self.size])
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        """
        only do it for image
        """
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Pad(object):
    def __init__(self, padding_n, padding_fill_value=0, padding_fill_target_value=0):
        self.padding_n = padding_n
        self.padding_fill_value = padding_fill_value
        self.padding_fill_target_value = padding_fill_target_value

    def __call__(self, image, target):
        image = F.pad(image, self.padding_n, self.padding_fill_value)
        if target is not None:
            target = F.pad(target, self.padding_n, self.padding_fill_target_value)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        if target is not None:
            target = torch.as_tensor(np.array(target), dtype=torch.long)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
