import torch
import torch.nn as nn

from ignite.utils import to_onehot


class BinaryJaccardWithLogitsLoss(nn.Module):

    def __init__(self, reduction=None):
        super(BinaryJaccardWithLogitsLoss, self).__init__()
        if isinstance(reduction, str):
            assert reduction in ('mean', 'sum')
        self.reduction = reduction

    def forward(self, y_pred, y):
        y_pred = torch.sigmoid(y_pred)

        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        y = y.reshape(y.shape[0], -1)

        intersection = y_pred * y
        union = y_pred + y - intersection + 1e-10

        intersection = torch.sum(intersection, dim=1)
        union = torch.sum(union, dim=1)

        if self.reduction == "mean":
            intersection = torch.mean(intersection, dim=0)
            union = torch.mean(union, dim=0)
        elif self.reduction == 'sum':
            intersection = torch.sum(intersection, dim=0)
            union = torch.sum(union, dim=0)

        return 1.0 - intersection / union


class SoftmaxJaccardWithLogitsLoss(nn.Module):

    def __init__(self, reduction=None, ignore_index=None):
        super(SoftmaxJaccardWithLogitsLoss, self).__init__()
        if isinstance(reduction, str):
            assert reduction in ('mean', 'sum')
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, y_pred, y):
        y_pred = torch.softmax(y_pred, dim=1)

        b, c = y_pred.shape[0], y_pred.shape[1]
        if y_pred.ndim != y.ndim:
            input_shape = y_pred.shape
            input_shape = (input_shape[0], input_shape[2], input_shape[3])
            if input_shape == y.shape:
                y = to_onehot(y, num_classes=c).to(y_pred)
            else:
                raise ValueError("Shapes mismatch: {} vs {}".format(y_pred.shape, y.shape))

        y_pred = y_pred.reshape(b, c, -1)
        y = y.reshape(b, c, -1)

        intersection = y_pred * y
        union = y_pred + y - intersection + 1e-10

        intersection = torch.sum(intersection, dim=-1)
        union = torch.sum(union, dim=-1)

        if self.ignore_index is not None:
            indices = list(range(c))
            indices.remove(self.ignore_index)
            intersection = intersection[:, indices]
            union = union[:, indices]

        if self.reduction == "mean":
            intersection = torch.mean(intersection)
            union = torch.mean(union)
        elif self.reduction == "sum":
            intersection = torch.sum(intersection)
            union = torch.sum(union)

        return 1.0 - intersection / union
