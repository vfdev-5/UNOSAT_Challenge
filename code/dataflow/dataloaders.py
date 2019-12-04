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


def get_train_mean_std(train_dataset, unique_id="", cache_dir="/tmp/unosat/"):
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
    
    if len(unique_id) > 0:
        unique_id += "_"

    fp = cache_dir / "train_mean_std_{}{}.pth".format(len(train_dataset), unique_id)
    
    if fp.exists():
        mean_std = torch.load(fp.as_posix())
    else:
        from ignite.engine import Engine
        from ignite.metrics import VariableAccumulation, Average
        from ignite.contrib.handlers import ProgressBar
        from albumentations.pytorch import ToTensorV2

        # Until https://github.com/pytorch/ignite/pull/681 is not merged
        class _Average(Average):
            def __init__(self, output_transform=lambda x: x, device=None):                
                super(_Average, self).__init__(output_transform=output_transform, device=device)                
                def _mean_op(a, x):
                    if x.ndim > 1:
                        x = x.sum(dim=0)            
                    return a + x            
                self._op = _mean_op

        train_dataset = TransformedDataset(train_dataset, transform_fn=ToTensorV2())
        train_loader = DataLoader(train_dataset, shuffle=False, drop_last=False, batch_size=16, num_workers=10, pin_memory=False)
        
        def compute_mean_std(engine, batch):
            b, c, *_ = batch['image'].shape
            data = batch['image'].reshape(b, c, -1).to(dtype=torch.float64)
            mean = torch.mean(data, dim=-1)
            mean2 = torch.mean(data ** 2, dim=-1)
            
            return {
                "mean": mean,
                "mean^2": mean2,
            }

        compute_engine = Engine(compute_mean_std)
        ProgressBar(desc="Compute Mean/Std").attach(compute_engine)
        img_mean = _Average(output_transform=lambda output: output['mean'])
        img_mean2 = _Average(output_transform=lambda output: output['mean^2'])
        img_mean.attach(compute_engine, 'mean')
        img_mean2.attach(compute_engine, 'mean2')
        state = compute_engine.run(train_loader)
        state.metrics['std'] = torch.sqrt(state.metrics['mean2'] - state.metrics['mean'] ** 2)        
        mean_std = {'mean': state.metrics['mean'], 'std': state.metrics['std']}
        torch.save(mean_std, fp.as_posix())
    
    return mean_std['mean'].tolist(), mean_std['std'].tolist()
