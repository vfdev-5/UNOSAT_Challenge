# Inference with baseline segmentation LWRefineNet
import os
from functools import partial

import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import FPN

import ttach as tta

from dataflow.datasets import get_trainval_datasets, read_img_in_db_with_mask
from dataflow.dataloaders import get_train_val_loaders
from dataflow.transforms import inference_prepare_batch_f32, denormalize, TransformedDataset


#################### Globals ####################

seed = 12
debug = False
device = 'cuda'

num_classes = 2

#################### Dataflow ####################

assert "INPUT_PATH" in os.environ
data_path = os.path.join(os.environ['INPUT_PATH'], "train_tiles")
csv_path = os.path.join(data_path, "tile_stats.csv")

train_folds = [0, 1, 2]
val_folds = [3, ]

train_ds, val_ds = get_trainval_datasets(data_path, csv_path, train_folds=train_folds, val_folds=val_folds,
                                         read_img_mask_fn=read_img_in_db_with_mask)


batch_size = 32
num_workers = 12

mean = [-17.398721187929123, -10.020421713800838, -12.10841437771272]
std = [6.290316422115964, 5.776936185931195, 5.795418280085563]
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

model = FPN(encoder_name='se_resnext50_32x4d', classes=2, encoder_weights=None)
run_uuid = "5230c20f609646cb9870a211036ea5cb"
weights_filename = "best_model_67_val_miou_bg=0.7574240313552584.pth"

has_targets = True

tta_transforms = tta.Compose([
    tta.Rotate90(angles=[90, -90, 180]),
])
