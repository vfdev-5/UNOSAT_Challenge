from typing import Type, Callable, Sequence

import numpy as np

import torch
from torch.utils.data import Dataset

from ignite.utils import convert_tensor


class TransformedDataset(Dataset):

    def __init__(self, ds: Type[Dataset], transform_fn: Callable):
        assert isinstance(ds, Dataset)
        assert callable(transform_fn)
        self.ds = ds
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        dp = self.ds[index]
        return self.transform_fn(**dp)


def denormalize(t: Type[torch.Tensor],
                mean: Sequence,
                std: Sequence,
                max_pixel_value: float = 255.0):
    assert isinstance(t, torch.Tensor), "{}".format(type(t))
    assert t.ndim == 3
    assert len(mean) == len(std) == t.shape[0], "{} vs {} vs {}".format(len(mean), len(std), t.shape[0])
    d = t.device
    mean = torch.tensor(mean, device=d).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(std, device=d).unsqueeze(-1).unsqueeze(-1)
    tensor = std * t + mean
    tensor *= max_pixel_value
    return tensor


def prepare_batch_fp32(batch, device, non_blocking):
    x, y = batch['image'], batch['mask']
    x = convert_tensor(x, device, non_blocking=non_blocking)
    y = convert_tensor(y, device, non_blocking=non_blocking).long()
    return x, y


def inference_prepare_batch_f32(batch, device, non_blocking):
    x = batch['image']
    y = batch['mask'] if 'mask' in batch else None
    meta = batch['meta'] if 'meta' in batch else None

    x = convert_tensor(x, device, non_blocking=non_blocking)
    if y is not None:
        y = convert_tensor(y, device, non_blocking=non_blocking).long()
    return x, y, meta
