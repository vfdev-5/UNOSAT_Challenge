# Inference with baseline segmentation LWRefineNet
import os
from functools import partial

import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import FPN

import ttach as tta

from dataflow.datasets import UnoSatTestTiles, read_img_in_db
from dataflow.dataloaders import get_inference_dataloader
from dataflow.transforms import inference_prepare_batch_f32, denormalize, TransformedDataset


#################### Globals ####################

seed = 12
debug = False
device = 'cuda'

num_classes = 2

#################### Dataflow ####################

assert "INPUT_PATH" in os.environ
data_path = os.path.join(os.environ['INPUT_PATH'], "test_tiles")

test_dataset = UnoSatTestTiles(data_path)
test_dataset = TransformedDataset(test_dataset, transform_fn=read_img_in_db)

batch_size = 4
num_workers = 12

mean = [-17.398721187929123, -10.020421713800838, -12.10841437771272]
std = [6.290316422115964, 5.776936185931195, 5.795418280085563]
max_value = 1.0


transforms = A.Compose([
    A.Normalize(mean=mean, std=std, max_pixel_value=max_value),
    ToTensorV2()
])

data_loader = get_inference_dataloader(
    test_dataset,
    transforms=transforms,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
)

prepare_batch = inference_prepare_batch_f32

# Image denormalization function to plot predictions with images
img_denormalize = partial(denormalize, mean=mean, std=std)

#################### Model ####################

model = FPN(encoder_name='se_resnext50_32x4d', classes=2, encoder_weights=None)
run_uuid = "5230c20f609646cb9870a211036ea5cb"
weights_filename = "best_model_67_val_miou_bg=0.7574240313552584.pth"

# TTA
tta_transforms = tta.Compose([
    tta.Rotate90(angles=[90, -90, 180]),
])
