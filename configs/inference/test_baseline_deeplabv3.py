# Inference with baseline segmentation DeeplabV3
import os
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from torchvision.models.segmentation import deeplabv3_resnet101

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataflow.datasets import UnoSatTestTiles, read_img_only
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
test_dataset = TransformedDataset(test_dataset, transform_fn=read_img_only)

batch_size = 2

num_workers = 12

mean = (0.0, 0.0, 0.0)
std = (5.0, 5.0, 5.0)
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

model = deeplabv3_resnet101(num_classes=num_classes)
run_uuid = "b0ae05ccbdca4d37b3455da300de7983"
weights_filename = "best_model_1_val_f1=0.8130922.pth"

def model_output_transform(y_pred):
    return y_pred['out']
