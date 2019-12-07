# Baseline segmentation
import os
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataflow.datasets import get_trainval_datasets, read_img_in_db_with_mask
from dataflow.dataloaders import get_train_val_loaders, get_train_sampler, get_train_mean_std
from dataflow.transforms import prepare_batch_fp32, denormalize

from models import LWRefineNet
from losses import SumOfLosses
from losses.jaccard import SoftmaxJaccardWithLogitsLoss

#################### Globals ####################

seed = 12
debug = False
device = 'cuda'
fp16_opt_level = "O2"

num_classes = 2
val_interval = 1

start_by_validation = False

#################### Dataflow ####################

assert "INPUT_PATH" in os.environ
data_path = os.path.join(os.environ['INPUT_PATH'], "train_tiles")
csv_path = os.path.join(data_path, "tile_stats.csv")

train_folds = [0, 1, 3]
val_folds = [2, ]

train_ds, val_ds = get_trainval_datasets(data_path, csv_path, train_folds=train_folds, val_folds=val_folds, read_img_mask_fn=read_img_in_db_with_mask)

train_sampler = get_train_sampler(train_ds, weight_per_class=(0.5, 0.5))
mean, std = get_train_mean_std(train_ds, unique_id="3b_in_db")
print("Computed mean/std: {}/{}".format(mean, std))

batch_size = 24
num_workers = 12
val_batch_size = 24

# According to https://arxiv.org/pdf/1906.06423.pdf
# For example: Train size: 224 -> Test size: 320 = max accuracy on ImageNet with ResNet-50
val_img_size = 512
train_img_size = 480

max_value = 1.0

train_transforms = A.Compose([
    A.RandomResizedCrop(train_img_size, train_img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
    A.OneOf([
        A.RandomRotate90(),
        A.Flip(),
    ]),
    A.Normalize(mean=mean, std=std, max_pixel_value=max_value),
    ToTensorV2()
])


val_transforms = A.Compose([
    A.Normalize(mean=mean, std=std, max_pixel_value=max_value),
    ToTensorV2()
])


train_loader, val_loader, train_eval_loader = get_train_val_loaders(
    train_ds, val_ds,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    batch_size=batch_size,
    num_workers=num_workers,
    val_batch_size=val_batch_size,
    pin_memory=True,
    train_sampler=train_sampler,
    limit_train_num_samples=100 if debug else None,
    limit_val_num_samples=100 if debug else None
)

accumulation_steps = 2

prepare_batch = prepare_batch_fp32

# Image denormalization function to plot predictions with images
img_denormalize = partial(denormalize, mean=mean, std=std)

#################### Model ####################

model = LWRefineNet(num_channels=3, num_classes=num_classes)

#################### Solver ####################

num_epochs = 50

names = ["cross entropy loss", "jaccard loss"]
xentropy = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]))
jaccard_loss = SoftmaxJaccardWithLogitsLoss()
criterion = SumOfLosses([xentropy, jaccard_loss], coeffs=[1.0, 2.0], names=names, total_loss_name="supervised batch loss")

output_names = names + ["supervised batch loss", ]

lr = 0.001
weight_decay = 1e-4
optimizer = optim.Adam(model.parameters(), lr=1.0, weight_decay=weight_decay)


le = len(train_loader)


def lambda_lr_scheduler(iteration, lr0, n, a):
    return lr0 * pow((1.0 - 1.0 * iteration / n), a)


lr_scheduler = lrs.LambdaLR(optimizer, lr_lambda=partial(lambda_lr_scheduler, lr0=lr, n=num_epochs * le, a=0.9))