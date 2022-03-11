#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
__all__ = ["BinaryLoss", "CrossEntropyLoss", "DiceLoss", "FocalLoss"]

import torch
import torch.nn.functional as F
from torch import nn
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from DRSegFL import utils


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor or None): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float or None): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        # if reduction is "none", then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError("avg_factor can not be used with reduction='sum'")
    return loss


class BinaryLoss(nn.Module):
    def __init__(self, loss_type="ce", reduction="mean", class_weight=None, class_weight_norm=False, loss_weight=1.0, smooth=1.0):
        super(BinaryLoss, self).__init__()
        assert loss_type in ["ce", "dice", "cbce", "ce_dice"]
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.class_weight_norm = class_weight_norm
        self.loss_type = loss_type
        self.smooth = smooth

    def forward(self, pred, label, weight=None, avg_factor=None, reduction_override=None):
        """
        :param pred: [N,C,*]
        :param label: [N,*]
        :param weight:
        :param avg_factor:
        :param reduction_override:
        :return:
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = (reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
            assert class_weight.shape[0] == pred.shape[1], \
                "Expect class weight shape [{}], get[{}]".format(pred.shape[1], class_weight.shape[0])
        else:
            class_weight = None

        loss_func = None
        if self.loss_type == "ce":
            loss_func = self.binary_ce_loss
        elif self.loss_type == "dice":
            loss_func = self.binary_dice_loss
        elif self.loss_type == "ce_dice":
            loss_func = self.binary_ce_dice_loss
        elif self.loss_type == "cbce":
            loss_func = self.binary_cbce_loss

        loss_cls = self.loss_weight * self.binary_loss(
            pred,
            label,
            loss_func,
            weight,
            class_weight=class_weight,
            class_weight_norm=self.class_weight_norm,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth
        )
        return loss_cls

    @classmethod
    def binary_ce_loss(cls, pred, label):
        """
        :param pred: [N, *]: here should be scores in [0,1]
        :param label: [N, *]: values is 0 or 1
        :return: [N]
        """
        loss = F.binary_cross_entropy(pred, label, reduction="none")
        loss = torch.mean(loss, dim=(1, 2))
        return loss

    @classmethod
    def binary_cbce_loss(cls, pred, label):
        """
        :param pred: [N, *]: here should be scores in [0,1]
        :param label: [N, *]: values is 0 or 1
        :return: [N]
        """
        mask = (label > 0.5).float()
        b, h, w = mask.shape
        num_pos = torch.sum(mask, dim=[1, 2]).float()  # Shape: [N,].
        num_neg = h * w - num_pos  # Shape: [N,].
        weight = torch.zeros_like(mask)
        pos_weight = num_neg / (num_pos + num_neg)
        neg_weight = num_pos / (num_pos + num_neg)
        for i in range(b):
            weight[i][label[i] > 0.5] = pos_weight[i]
            weight[i][label[i] <= 0.5] = neg_weight[i]
        loss = torch.nn.functional.binary_cross_entropy(pred.float(), label.float(), weight=weight, reduction="none")
        return loss

    @classmethod
    def binary_dice_loss(cls, pred, label, smooth=1.0):
        """
        :param pred: [N, *]: here should be scores in [0,1]
        :param label: [N, *]: values is 0 or 1
        :param smooth: smooth
        :return: [N]
        """

        pred = pred.contiguous().view(pred.shape[0], -1).float()
        label = label.contiguous().view(label.shape[0], -1).float()

        num = 2 * torch.sum(torch.mul(pred, label), dim=1) + smooth
        den = torch.sum(pred, dim=1) + torch.sum(label, dim=1) + smooth

        loss = 1. - num / den

        return loss

    @classmethod
    def binary_ce_dice_loss(cls, pred, label, smooth=1.0):
        loss1 = cls.binary_ce_loss(pred, label)
        loss2 = cls.binary_dice_loss(pred, label, smooth=smooth)

        return loss1 + loss2

    @classmethod
    def binary_loss(cls, pred_raw, label_raw,
                    loss_f, weight, class_weight, class_weight_norm=False, reduction="mean", avg_factor=None, smooth=1.0):
        """
            :param pred:  [N, C, *] scores without softmax
            :param label: [N, *] in [0, C], 0 stands for background, 1~C stands for pred in 0~C-1
            :return: reduction([N])
        """
        pred = pred_raw.clone()
        label = label_raw.clone()
        num_classes = pred.shape[1]
        if class_weight is not None:
            class_weight = class_weight.float()

        if pred.shape != label.shape:
            label = utils.batch_make_one_hot(label, num_classes)  # [N,C,*]

        pred = torch.sigmoid(pred)

        loss = 0.
        for i in range(num_classes):
            if isinstance(loss_f, tuple):
                loss_function = loss_f[i]
            else:
                loss_function = loss_f
            class_loss = loss_function(pred[:, i], label[:, i], smooth=smooth)
            if class_weight is not None:
                class_loss *= class_weight[i]
            loss += class_loss

        if class_weight is not None and class_weight_norm:
            loss = loss / torch.sum(class_weight)
        else:
            loss = loss / num_classes
        loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return loss


class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        reduction (str, optional): . Defaults to "mean".
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, use_sigmoid=False, reduction="mean", class_weight=None, loss_weight=1.0, ignore_index=255):
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index

        if self.use_sigmoid:
            self.loss_f = self.binary_cross_entropy
        else:
            self.loss_f = self.cross_entropy

    def forward(self, pred, label, weight=None, avg_factor=None, reduction_override=None):
        """
        :param pred: [N,C,*]
        :param target: [N,*] values in [0,num_classes)
        :param avg_factor:
        :param reduction_override:
        :return:
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = (reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.loss_f(
            pred,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=self.ignore_index)
        return loss_cls

    @classmethod
    def cross_entropy(cls, pred, label, weight=None, class_weight=None, reduction="mean", avg_factor=None, ignore_index=255):
        """The wrapper function for :func:`F.cross_entropy`"""
        # class_weight is a manual rescaling weight given to each class.
        # If given, has to be a Tensor of size C element-wise losses
        loss = F.cross_entropy(pred, label, weight=class_weight, reduction="none", ignore_index=ignore_index)

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        return loss

    @classmethod
    def binary_cross_entropy(cls, pred, label, weight=None, reduction="mean", avg_factor=None, class_weight=None, ignore_index=255):
        """Calculate the binary CrossEntropy loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, 1).
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            class_weight (list[float], optional): The weight for each class.
            ignore_index (int | None): The label index to be ignored. Default: 255

        Returns:
            torch.Tensor: The calculated loss
        """
        if pred.dim() != label.dim():
            assert (pred.dim() == 2 and label.dim() == 1) or (pred.dim() == 4 and label.dim() == 3), \
                "Only pred shape [N, C], label shape [N] or pred shape [N, C, H, W], label shape [N, H, W] are supported"
            label, weight = cls._expand_onehot_labels(label, weight, pred.shape, ignore_index)

        # weighted element-wise losses
        if weight is not None:
            weight = weight.float()
        loss = F.binary_cross_entropy_with_logits(pred, label.float(), pos_weight=class_weight, reduction="none")
        # do the reduction for the weighted loss
        loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)

        return loss

    @classmethod
    def _expand_onehot_labels(cls, labels, label_weights, target_shape, ignore_index):
        """Expand onehot labels to match the size of prediction."""
        bin_labels = labels.new_zeros(target_shape)
        valid_mask = (labels >= 0) & (labels != ignore_index)
        inds = torch.nonzero(valid_mask, as_tuple=True)

        if inds[0].numel() > 0:
            if labels.dim() == 3:
                bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
            else:
                bin_labels[inds[0], labels[valid_mask]] = 1

        valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
        if label_weights is None:
            bin_label_weights = valid_mask
        else:
            bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
            bin_label_weights *= valid_mask

        return bin_labels, bin_label_weights


class DiceLoss(nn.Module):
    """DiceLoss.

    This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

    Args:
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: "mean".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to "loss_dice".
    """

    def __init__(self, smooth=1, exponent=2, reduction="mean", class_weight=None, loss_weight=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, avg_factor=None, reduction_override=None):
        """
        :param pred: [N,C,*]
        :param target: [N,*] values in [0,num_classes)
        :param avg_factor:
        :param reduction_override:
        :return:
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * self.dice_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        return loss

    @classmethod
    def dice_loss(cls, pred, target, valid_mask,
                  weight=None, reduction="mean", avg_factor=None, smooth=1, exponent=2, class_weight=None, ignore_index=255):
        assert pred.shape[0] == target.shape[0]
        total_loss = 0
        num_classes = pred.shape[1]
        for i in range(num_classes):
            if i != ignore_index:
                dice_loss = cls.binary_dice_loss(pred[:, i], target[..., i], valid_mask=valid_mask, smooth=smooth, exponent=exponent)
                if class_weight is not None:
                    dice_loss *= class_weight[i]
                total_loss += dice_loss
        loss = total_loss / num_classes
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    @classmethod
    def binary_dice_loss(cls, pred, target, valid_mask, weight=None, reduction="mean", avg_factor=None, smooth=1, exponent=2):
        assert pred.shape[0] == target.shape[0]
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

        num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
        den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

        loss = 1 - num / den
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

        return loss


class FocalLoss(nn.Module):
    def __init__(self, use_sigmoid=True, gamma=2.0, alpha=0.5, reduction="mean", class_weight=None, loss_weight=1.0, loss_name="loss_focal",
                 ignore_index=255):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float | list[float], optional): A balanced form for Focal
                Loss. Defaults to 0.5. When a list is provided, the length
                of the list should be equal to the number of classes.
                Please be careful that this parameter is not the
                class-wise weight but the weight of a binary classification
                problem. This binary classification problem regards the
                pixels which belong to one class as the foreground
                and the other pixels as the background, each element in
                the list is the weight of the corresponding foreground class.
                The value of alpha or each element of alpha should be a float
                in the interval [0, 1]. If you want to specify the class-wise
                weight, please use `class_weight` parameter.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to "mean". Options are "none", "mean" and
                "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            loss_name (str, optional): Name of the loss item. If you want this
                loss item to be included into the backward graph, `loss_` must
                be the prefix of the name. Defaults to "loss_focal".
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, \
            "AssertionError: Only sigmoid focal loss supported now."
        assert reduction in ("none", "mean", "sum"), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert isinstance(alpha, (float, list)), \
            "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), \
            "AssertionError: gamma should be of type float"
        assert isinstance(loss_weight, float), \
            "AssertionError: loss_weight should be of type float"
        assert isinstance(loss_name, str), \
            "AssertionError: loss_name should be of type str"
        assert isinstance(class_weight, list) or class_weight is None, \
            "AssertionError: class_weight must be None or of type list"
        assert isinstance(ignore_index, int), \
            "ignore_index must be of type int"
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.ignore_index = ignore_index

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction with shape
                (N, C) where C = number of classes, or
                (N, C, d_1, d_2, ..., d_K) with K≥1 in the
                case of K-dimensional loss.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1,
                or (N, d_1, d_2, ..., d_K) with K≥1 in the case of
                K-dimensional loss. If containing class probabilities,
                same shape as the input.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to
                average the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used
                to override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
            ignore_index (int, optional): The label index to be ignored.
                Default: 255
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, "none", "mean", "sum"), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert pred.shape == target.shape or \
               (pred.size(0) == target.size(0) and
                pred.shape[2:] == target.shape[1:]), \
            "The shape of pred doesn't match the shape of target"

        original_shape = pred.shape

        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()

        if original_shape == target.shape:
            # target with shape [B, C, d_1, d_2, ...]
            # transform it's shape into [N, C]
            # [B, C, d_1, d_2, ...] -> [C, B, d_1, d_2, ..., d_k]
            target = target.transpose(0, 1)
            # [C, B, d_1, d_2, ..., d_k] -> [C, N]
            target = target.reshape(target.size(0), -1)
            # [C, N] -> [N, C]
            target = target.transpose(0, 1).contiguous()
        else:
            # target with shape [B, d_1, d_2, ...]
            # transform it's shape into [N, ]
            target = target.view(-1).contiguous()
            valid_mask = (target != self.ignore_index).view(-1, 1)
            # avoid raising error when using F.one_hot()
            target = torch.where(target == self.ignore_index, target.new_tensor(0),
                                 target)

        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            num_classes = pred.size(1)
            if torch.cuda.is_available() and pred.is_cuda:
                if target.dim() == 1:
                    one_hot_target = F.one_hot(target, num_classes=num_classes)
                else:
                    one_hot_target = target
                    target = target.argmax(dim=1)
                    valid_mask = (target != self.ignore_index).view(-1, 1)
                calculate_loss_func = self.sigmoid_focal_loss
            else:
                one_hot_target = None
                if target.dim() == 1:
                    target = F.one_hot(target, num_classes=num_classes)
                else:
                    valid_mask = (target.argmax(dim=1) != self.ignore_index).view(
                        -1, 1)
                calculate_loss_func = self.py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                one_hot_target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                class_weight=self.class_weight,
                valid_mask=valid_mask,
                reduction=reduction,
                avg_factor=avg_factor)

            if reduction == "none":
                # [N, C] -> [C, N]
                loss_cls = loss_cls.transpose(0, 1)
                # [C, N] -> [C, B, d1, d2, ...]
                # original_shape: [B, C, d1, d2, ...]
                loss_cls = loss_cls.reshape(original_shape[1],
                                            original_shape[0],
                                            *original_shape[2:])
                # [C, B, d1, d2, ...] -> [B, C, d1, d2, ...]
                loss_cls = loss_cls.transpose(0, 1).contiguous()
        else:
            raise NotImplementedError
        return loss_cls

    @classmethod
    def py_sigmoid_focal_loss(cls, pred, target,
                              one_hot_target=None, weight=None, gamma=2.0, alpha=0.5, class_weight=None, valid_mask=None,
                              reduction="mean", avg_factor=None):
        """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

        Args:
            pred (torch.Tensor): The prediction with shape (N, C), C is the
                number of classes
            target (torch.Tensor): The learning label of the prediction with
                shape (N, C)
            one_hot_target (None): Placeholder. It should be None.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float | list[float], optional): A balanced form for Focal Loss.
                Defaults to 0.5.
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            valid_mask (torch.Tensor, optional): A mask uses 1 to mark the valid
                samples and uses 0 to mark the ignored samples. Default: None.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to "mean".
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        """
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) *
                        (1 - target)) * one_minus_pt.pow(gamma)

        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none") * focal_weight
        final_weight = torch.ones(1, pred.size(1)).type_as(loss)
        if weight is not None:
            if weight.shape != loss.shape and weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (N, ),
                # which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            assert weight.dim() == loss.dim()
            final_weight = final_weight * weight
        if class_weight is not None:
            final_weight = final_weight * pred.new_tensor(class_weight)
        if valid_mask is not None:
            final_weight = final_weight * valid_mask
        loss = weight_reduce_loss(loss, final_weight, reduction, avg_factor)
        return loss

    @classmethod
    def sigmoid_focal_loss(cls, pred, target,
                           one_hot_target, weight=None, gamma=2.0, alpha=0.5, class_weight=None, valid_mask=None,
                           reduction="mean", avg_factor=None):
        r"""A warpper of cuda version `Focal Loss
        <https://arxiv.org/abs/1708.02002>`_.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C), C is the number
                of classes.
            target (torch.Tensor): The learning label of the prediction. It's shape
                should be (N, )
            one_hot_target (torch.Tensor): The learning label with shape (N, C)
            weight (torch.Tensor, optional): Sample-wise loss weight.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float | list[float], optional): A balanced form for Focal Loss.
                Defaults to 0.5.
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            valid_mask (torch.Tensor, optional): A mask uses 1 to mark the valid
                samples and uses 0 to mark the ignored samples. Default: None.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to "mean". Options are "none", "mean" and "sum".
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        """
        # Function.apply does not accept keyword arguments, so the decorator
        # "weighted_loss" is not applicable
        final_weight = torch.ones(1, pred.size(1)).type_as(pred)
        if isinstance(alpha, list):
            # _sigmoid_focal_loss doesn't accept alpha of list type. Therefore, if
            # a list is given, we set the input alpha as 0.5. This means setting
            # equal weight for foreground class and background class. By
            # multiplying the loss by 2, the effect of setting alpha as 0.5 is
            # undone. The alpha of type list is used to regulate the loss in the
            # post-processing process.
            loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(),
                                       gamma, 0.5, None, "none") * 2
            alpha = pred.new_tensor(alpha)
            final_weight = final_weight * (
                    alpha * one_hot_target + (1 - alpha) * (1 - one_hot_target))
        else:
            loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(),
                                       gamma, alpha, None, "none")
        if weight is not None:
            if weight.shape != loss.shape and weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (N, ),
                # which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            assert weight.dim() == loss.dim()
            final_weight = final_weight * weight
        if class_weight is not None:
            final_weight = final_weight * pred.new_tensor(class_weight)
        if valid_mask is not None:
            final_weight = final_weight * valid_mask
        loss = weight_reduce_loss(loss, final_weight, reduction, avg_factor)
        return loss
