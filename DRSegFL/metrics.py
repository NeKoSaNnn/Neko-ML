#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import numpy as np


def intersect_and_union(pred_label, label, num_classes, ignore_index):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction segmentation map
        label (ndarray): Ground truth segmentation map
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes
         ndarray: The union of prediction and ground truth histogram on all
             classes
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def mIoU(results, gt_seg_maps, num_classes, ignore_index, nan_to_num=None):
    """Calculate Intersection and Union (IoU)

    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, )
         ndarray: Per category IoU, shape (num_classes, )
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs, "{}!={}".format(len(gt_seg_maps), num_imgs)
    total_area_intersect = np.zeros((num_classes,), dtype=np.float)
    total_area_union = np.zeros((num_classes,), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes,), dtype=np.float)
    total_area_label = np.zeros((num_classes,), dtype=np.float)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                                                                      ignore_index=ignore_index)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    accs = total_area_intersect / total_area_label
    ious = total_area_intersect / total_area_union
    if nan_to_num is not None:
        return all_acc, np.nan_to_num(accs, nan=nan_to_num), np.nan_to_num(ious, nan=nan_to_num)
    return all_acc, accs, ious


def Dice2IoU(dice):
    if dice is None:
        return None
    return dice / (2 - dice)


def IoU2Dice(iou):
    if iou is None:
        return None
    return 2 * iou / (1 + iou)


def dice_coeff(preds, targets, epsilon=1e-6):
    assert preds.shape == targets.shape, "{}!={}".format(preds.shape, targets.shape)
    pred = preds.flatten(1)
    truth = targets.flatten(1)
    intersection = (pred * truth).double().sum(1)
    dice = (2.0 * intersection + epsilon) / (pred.double().sum(1) + truth.double().sum(1) + epsilon)
    return dice.mean()


def multi_dice_coeff(preds, targets, epsilon=1e-6):
    assert preds.shape == targets.shape, "{}!={}".format(preds.shape, targets.shape)
    num_classes = preds.shape[1]
    dice = 0
    for i_class in range(num_classes):
        dice += dice_coeff(preds[:, i_class, ...], targets[:, i_class, ...], epsilon)
    return dice / num_classes


def dice_loss(preds, targets, multiclass=True, epsilon=1e-6):
    assert preds.shape == targets.shape, "{}!={}".format(preds.shape, targets.shape)
    return 1 - multi_dice_coeff(preds, targets, epsilon) if multiclass else 1 - dice_coeff(preds, targets, epsilon)
