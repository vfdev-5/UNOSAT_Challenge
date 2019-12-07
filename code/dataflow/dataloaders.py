from typing import Type, Callable, Optional, Tuple, Union
from pathlib import Path

import numpy as np

import tqdm
import torch
import torch.distributed as dist
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataset import Dataset, Subset
import torch.utils.data.distributed as data_dist

from dataflow.transforms import TransformedDataset


def get_train_sampler(train_dataset, weight_per_class, cache_dir="/tmp/unosat/"):

    # Ensure that only process 0 in distributed performs the computation, and the others will use the cache
    if dist.get_rank() > 0:
        torch.distributed.barrier()  # synchronization point for all processes > 0

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

        if dist.get_rank() < 1:
            torch.save(weights, fp.as_posix())

    if dist.get_rank() < 1:
        torch.distributed.barrier()  # synchronization point for process 0

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
                          train_sampler: Optional[Sampler] = None,
                          val_sampler: Optional[Sampler] = None,
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

    if dist.is_available():
        if train_sampler is not None:
            train_sampler = DistributedProxySampler(train_sampler)
        else:
            train_sampler = data_dist.DistributedSampler(train_ds, shuffle=True)

        if val_sampler is not None:
            val_sampler = DistributedProxySampler(val_sampler)
        else:
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

    sampler = None
    if dist.is_available():
        sampler = data_dist.DistributedSampler(dataset, shuffle=False)

    loader = DataLoader(dataset, shuffle=False,
                        batch_size=batch_size, num_workers=num_workers,
                        sampler=sampler, pin_memory=pin_memory, drop_last=False)
    return loader


def get_train_mean_std(train_dataset, unique_id="", cache_dir="/tmp/unosat/"):
    # # Ensure that only process 0 in distributed performs the computation, and the others will use the cache
    # if dist.get_rank() > 0:
    #     torch.distributed.barrier()  # synchronization point for all processes > 0

    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
    
    if len(unique_id) > 0:
        unique_id += "_"

    fp = cache_dir / "train_mean_std_{}{}.pth".format(len(train_dataset), unique_id)
    
    if fp.exists():
        mean_std = torch.load(fp.as_posix())
    else:
        if dist.is_available() and dist.is_initialized():
            raise RuntimeError("Current implementation of Mean/Std computation is not working in distrib config")

        from ignite.engine import Engine
        from ignite.metrics import Average
        from ignite.contrib.handlers import ProgressBar
        from albumentations.pytorch import ToTensorV2

        train_dataset = TransformedDataset(train_dataset, transform_fn=ToTensorV2())
        train_loader = DataLoader(train_dataset, shuffle=False, drop_last=False,
                                  batch_size=16, num_workers=10, pin_memory=False)
        
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
        img_mean = Average(output_transform=lambda output: output['mean'])
        img_mean2 = Average(output_transform=lambda output: output['mean^2'])
        img_mean.attach(compute_engine, 'mean')
        img_mean2.attach(compute_engine, 'mean2')
        state = compute_engine.run(train_loader)
        state.metrics['std'] = torch.sqrt(state.metrics['mean2'] - state.metrics['mean'] ** 2)        
        mean_std = {'mean': state.metrics['mean'], 'std': state.metrics['std']}

        # if dist.get_rank() < 1:
        torch.save(mean_std, fp.as_posix())

    # if dist.get_rank() < 1:
    #     torch.distributed.barrier()  # synchronization point for process 0

    return mean_std['mean'].tolist(), mean_std['std'].tolist()


from torch.utils.data.distributed import DistributedSampler


# Waiting until https://github.com/pytorch/pytorch/issues/23430 to be closed
class DistributedProxySampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Input sampler is assumed to be of constant size.

    Arguments:
        sampler: Input data sampler.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, sampler, num_replicas=None, rank=None):        
        super(DistributedProxySampler, self).__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = sampler

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)