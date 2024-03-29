# Inference with baseline segmentation LWRefineNet
import os
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

import albumentations as A
from albumentations.pytorch import ToTensorV2

import ttach as tta

from dataflow.datasets import get_trainval_datasets
from dataflow.dataloaders import get_train_val_loaders
from dataflow.transforms import inference_prepare_batch_f32, denormalize, TransformedDataset

from models import LWRefineNet

#################### Globals ####################

seed = 12
debug = False
device = 'cuda'

num_classes = 2

#################### Dataflow ####################

assert "INPUT_PATH" in os.environ
data_path = os.path.join(os.environ['INPUT_PATH'], "train_tiles")
csv_path = os.path.join(data_path, "tile_stats.csv")

train_folds = [0, 1, 3]
val_folds = [2, ]

train_ds, val_ds = get_trainval_datasets(data_path, csv_path, train_folds=train_folds, val_folds=val_folds)


batch_size = 32
num_workers = 12

mean = (0.0, 0.0, 0.0)
std = (5.0, 5.0, 5.0)
max_value = 1.0


transforms = A.Compose([
    A.Normalize(mean=mean, std=std, max_pixel_value=max_value),
    ToTensorV2()
])

_, data_loader, _ = get_train_val_loaders(
    train_ds, val_ds,
    train_transforms=transforms,
    val_transforms=transforms,
    batch_size=batch_size,
    num_workers=num_workers,
    val_batch_size=batch_size,
    pin_memory=True,
)

prepare_batch = inference_prepare_batch_f32

# Image denormalization function to plot predictions with images
img_denormalize = partial(denormalize, mean=mean, std=std)

#################### Model ####################

model = LWRefineNet(num_channels=3, num_classes=num_classes)
run_uuid = "bf1fa0a668cd4d4da7de6f2c77b6bebb"
weights_filename = "checkpoint_model_28000.pth"


has_targets = True

tta_transforms = tta.Compose([
    tta.Rotate90(angles=[90, -90, 180]),
])
