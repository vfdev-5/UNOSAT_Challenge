from typing import Type, Callable, Optional, Tuple, Union
from pathlib import Path

import numpy as np

import tqdm
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataset import Dataset, Subset
import torch.utils.data.distributed as data_dist

from dataflow.transforms import TransformedDataset


def get_train_sampler(train_dataset, weight_per_class, cache_dir="/tmp/unosat/"):
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)

    fp = cache_dir / "train_sampler_weights_{}__{}.pth".format(len(train_dataset),
                                                               repr(weight_per_class).replace(", ", "_"))
    if fp.exists():
        weights = torch.load(fp.as_posix())
    else:
        weights = torch.tensor([0.0] * len(train_dataset))
        for i, dp in tqdm.tqdm(enumerate(train_dataset), total=len(train_dataset)):
            y = (dp['mask'] > 0).any()
            weights[i] = weight_per_class[y]
        torch.save(weights, fp.as_posix())

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler


def get_train_val_loaders(train_ds: Type[Dataset],
                          val_ds: Type[Dataset],
                          train_transforms: Callable,
                          val_transforms: Callable,
                          batch_size: int = 16,
                          num_workers: int = 8,
                          val_batch_size: Optional[int] = None,
                          pin_memory: bool = True,
                          train_sampler: Optional[Union[Sampler, str]] = None,
                          val_sampler: Optional[Union[Sampler, str]] = None,
                          limit_train_num_samples: Optional[int] = None,
                          limit_val_num_samples: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:

    if limit_train_num_samples is not None:
        np.random.seed(limit_train_num_samples)
        train_indices = np.random.permutation(len(train_ds))[:limit_train_num_samples]
        train_ds = Subset(train_ds, train_indices)

    if limit_val_num_samples is not None:
        np.random.seed(limit_val_num_samples)
        val_indices = np.random.permutation(len(val_ds))[:limit_val_num_samples]
        val_ds = Subset(val_ds, val_indices)

    # random samples for evaluation on training dataset
    if len(val_ds) < len(train_ds):
        train_eval_indices = np.random.permutation(len(train_ds))[:len(val_ds)]
        train_eval_ds = Subset(train_ds, train_eval_indices)
    else:
        train_eval_ds = train_ds

    train_ds = TransformedDataset(train_ds, transform_fn=train_transforms)
    val_ds = TransformedDataset(val_ds, transform_fn=val_transforms)
    train_eval_ds = TransformedDataset(train_eval_ds, transform_fn=val_transforms)

    if isinstance(train_sampler, str):
        assert train_sampler == 'distributed'
        train_sampler = data_dist.DistributedSampler(train_ds)

    if isinstance(val_sampler, str):
        assert val_sampler == 'distributed'
        # we shuffle validation for visualization purposes inside `predictions_gt_images_handler`
        val_sampler = data_dist.DistributedSampler(val_ds, shuffle=True)

    train_loader = DataLoader(train_ds, shuffle=train_sampler is None,
                              batch_size=batch_size, num_workers=num_workers,
                              sampler=train_sampler,
                              pin_memory=pin_memory, drop_last=True)

    val_batch_size = batch_size * 4 if val_batch_size is None else val_batch_size
    val_loader = DataLoader(val_ds, shuffle=False, sampler=val_sampler,
                            batch_size=val_batch_size, num_workers=num_workers,
                            pin_memory=pin_memory, drop_last=False)

    train_eval_loader = DataLoader(train_eval_ds, shuffle=False, sampler=val_sampler,
                                   batch_size=val_batch_size, num_workers=num_workers,
                                   pin_memory=pin_memory, drop_last=False)

    return train_loader, val_loader, train_eval_loader


def get_inference_dataloader(dataset: Type[Dataset],
                             transforms: Callable,
                             batch_size: int = 16,
                             num_workers: int = 8,
                             pin_memory: bool = True,
                             limit_num_samples: Optional[int] = None) -> DataLoader:

    if limit_num_samples is not None:
        np.random.seed(limit_num_samples)
        indices = np.random.permutation(len(dataset))[:limit_num_samples]
        dataset = Subset(dataset, indices)

    dataset = TransformedDataset(dataset, transform_fn=transforms)

    loader = DataLoader(dataset, shuffle=False,
                        batch_size=batch_size, num_workers=num_workers,
                        pin_memory=pin_memory, drop_last=False)
    return loader
