#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import torch
import torch.nn.functional as F


def slide_inference(imgs, model, num_classes: int, crop_size: int, stride: int):
    """Inference by sliding-window with overlap.

    If h_crop > h_img or w_crop > w_img, the small patch will be used to
    decode without padding.
    """
    assert isinstance(crop_size, int) or isinstance(crop_size, list) and len(crop_size) == 2
    assert isinstance(stride, int) or isinstance(stride, list) and len(stride) == 2
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    h_crop, w_crop = crop_size
    h_stride, w_stride = stride
    batch_size, _, h_img, w_img = imgs.size()  # (N,C,H,W)
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = imgs.new_zeros((batch_size, num_classes, h_img, w_img))
    count_mat = imgs.new_zeros((batch_size, 1, h_img, w_img))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = imgs[:, :, y1:y2, x1:x2]
            crop_seg_logit = model(crop_img)
            preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
            count_mat[:, :, y1:y2, x1:x2] += 1

    assert (count_mat == 0).sum() == 0
    preds = preds / count_mat
    return preds


def whole_inference(imgs, model):
    """Inference with full image."""

    preds = model(imgs)
    return preds
