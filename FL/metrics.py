#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""


def dice_coeff(input, target, epsilon=1e-6):
    assert input.shape == target.shape
    pred = input.flatten(1)
    truth = target.flatten(1)
    intersection = (pred * truth).double().sum(1)
    dice = (2.0 * intersection + epsilon) / (pred.double().sum(1) + truth.double().sum(1) + epsilon)
    return dice.mean()


def multi_dice_coeff(input, target, epsilon=1e-6):
    assert input.shape == target.shape
    num_classes = input.shape[1]
    dice = 0
    for i_class in range(num_classes):
        dice += dice_coeff(input[:, i_class, ...], target[:, i_class, ...], epsilon)
    return dice / num_classes


def dice_loss(input, target, multiclass=True, epsilon=1e-6):
    assert input.shape == target.shape
    return 1 - multi_dice_coeff(input, target, epsilon) if multiclass else 1 - dice_coeff(input, target, epsilon)
