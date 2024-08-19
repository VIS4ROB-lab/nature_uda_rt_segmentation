import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss


def _sensitivity_specificity_loss(y_true: torch.Tensor, y_pred: torch.Tensor, w: float = 0.5) -> torch.Tensor:
    """
    Sensitivity Specificity Loss.

    :param y_true: ground truth one-hot label
    :param y_pred: predicted logits
    :param w: weight for sensitivity

    :return: loss value
    """
    if y_true.shape != y_pred.shape:
        raise RuntimeError("y_true and y_pred must have the same shape")
    n_classes = y_true.shape[1]
    confusion_matrix = torch.zeros((n_classes, n_classes), dtype=torch.float)
    y_true = torch.argmax(y_true, dim=1)  # Reduce to [batch, h, w]
    y_pred = torch.argmax(y_pred, dim=1)
    # Use trick to compute the confusion matrix
    # Reference: https://github.com/monniert/docExtractor/
    for y_true_item, y_pred_item in zip(y_true, y_pred):
        y_true_item = y_true_item.flatten()  # Reduce to 1-D tensor
        y_pred_item = y_pred_item.flatten()
        confusion_matrix += torch.bincount(n_classes * y_true_item + y_pred_item, minlength=n_classes ** 2).reshape(
            n_classes, n_classes)
    # From confusion matrix, we compute tp, fp, fn, tn
    # Get the answer from this discussion:
    # https://stats.stackexchange.com/questions/179835/how-to-build-a-confusion-matrix-for-a-multiclass-classifier
    sum_along_classified = torch.sum(confusion_matrix, dim=1)  # sum(c1_1, cn_1) return 1D tensor
    sum_along_actual = torch.sum(confusion_matrix, dim=0)  # sum(c1_1 -> c1_n)
    tp = torch.diagonal(confusion_matrix, offset=0)
    fp = sum_along_classified - tp
    fn = sum_along_actual - tp
    tn = torch.ones(n_classes, dtype=torch.float) * torch.sum(confusion_matrix) - tp - fp - fn
    smooth = torch.ones(n_classes, dtype=torch.float) * 1e-6  # Use to avoid numeric division error
    assert tp.shape == fp.shape == fn.shape == tn.shape
    sensitivity = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    # Relation between tp, fp, fn, tn annotation vs set annotation here, so the actual loss become, compare this
    # loss vs the Soft Dice Loss, see https://arxiv.org/pdf/1803.11078.pdf
    return 1.0 - torch.mean(w * sensitivity + (1 - w) * specificity)


@LOSSES.register_module()
class SensitivitySpecificityLoss(nn.Module):
    def __init__(self, weight: float = 0.5):
        """
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        Args:
            weight: use for the combination of sensitivity and specificity
        """
        super(SensitivitySpecificityLoss, self).__init__()
        self.weight = weight

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        num_classes = output.shape[1]
        target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        output = F.softmax(output, dim=1)
        return _sensitivity_specificity_loss(target, output, self.weight)


# borrowed from mmsegmentation 1.1.1
@weighted_loss
def tversky_loss(pred,
                 target,
                 valid_mask,
                 alpha=0.3,
                 beta=0.7,
                 smooth=1,
                 class_weight=None,
                 ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            tversky_loss = binary_tversky_loss(
                pred[:, i],
                target[..., i],
                valid_mask=valid_mask,
                alpha=alpha,
                beta=beta,
                smooth=smooth)
            if class_weight is not None:
                tversky_loss *= class_weight[i]
            total_loss += tversky_loss
    return total_loss / num_classes


@weighted_loss
def binary_tversky_loss(pred,
                        target,
                        valid_mask,
                        alpha=0.3,
                        beta=0.7,
                        smooth=1):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    TP = torch.sum(torch.mul(pred, target) * valid_mask, dim=1)
    FP = torch.sum(torch.mul(pred, 1 - target) * valid_mask, dim=1)
    FN = torch.sum(torch.mul(1 - pred, target) * valid_mask, dim=1)
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return 1 - tversky


@LOSSES.register_module()
class TverskyLoss(nn.Module):
    """TverskyLoss. This loss is proposed in `Tversky loss function for image
    segmentation using 3D fully convolutional deep networks.

    <https://arxiv.org/abs/1706.05721>`_.
    Args:
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        alpha(float, in [0, 1]):
            The coefficient of false positives. Default: 0.3.
        beta (float, in [0, 1]):
            The coefficient of false negatives. Default: 0.7.
            Note: alpha + beta = 1.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_tversky'.
    """

    def __init__(self,
                 smooth=1,
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 alpha=0.3,
                 beta=0.7,
                 loss_name='loss_tversky'):
        super().__init__()
        self.smooth = smooth
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        assert (alpha + beta == 1.0), 'Sum of alpha and beta but be 1.0!'
        self.alpha = alpha
        self.beta = beta
        self._loss_name = loss_name

    def forward(self, pred, target, **kwargs):
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * tversky_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            alpha=self.alpha,
            beta=self.beta,
            smooth=self.smooth,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
