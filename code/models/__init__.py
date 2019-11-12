
import torch.nn as nn
from torch.nn import functional as F

from models.lwrefinenet import MBv2


class LWRefineNet(nn.Module):

    def __init__(self, *args, **kwargs):
        super(LWRefineNet, self).__init__()
        self.mbv2 = MBv2(*args, **kwargs)

    def forward(self, x):
        original_size = (x.shape[-2], x.shape[-1])
        y_pred = self.mbv2(x)
        return F.interpolate(y_pred, size=original_size, mode='bicubic', align_corners=False)
