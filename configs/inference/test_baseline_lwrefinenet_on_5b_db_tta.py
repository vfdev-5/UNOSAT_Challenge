# Inference with baseline segmentation LWRefineNet
import os
from functools import partial

import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

import ttach as tta

from dataflow.datasets import UnoSatTestTiles, read_img_5b_in_db
from dataflow.dataloaders import get_inference_dataloader
from dataflow.transforms import inference_prepare_batch_f32, denormalize, TransformedDataset

from models import LWRefineNet

#################### Globals ####################

seed = 12
debug = False
device = 'cuda'

num_classes = 2

#################### Dataflow ####################

assert "INPUT_PATH" in os.environ
data_path = os.path.join(os.environ['INPUT_PATH'], "test_tiles")

test_dataset = UnoSatTestTiles(data_path)
test_dataset = TransformedDataset(test_dataset, transform_fn=read_img_5b_in_db)

batch_size = 3
num_workers = 12

mean = [-17.704988005545587, -10.33310725243658, -12.422949109368183, 213.3866453581477, 0.4748089840110086]
std = [6.5437130712772795, 6.033536195001276, 6.063934363438651, 245.40096009414592, 238.8577452846451]
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
def img_denormalize(nimg): 
    img = denormalize(nimg, mean=mean, std=std)
    return img[(0, 1, 2), :, :]

#################### Model ####################

model = LWRefineNet(num_channels=5, num_classes=num_classes)
run_uuid = "c799f69b388c482da58d68a0bf7bb98e"
weights_filename = "best_model_47_val_miou_bg=0.7540399538484317.pth"


# TTA
tta_transforms = tta.Compose([
    tta.Rotate90(angles=[90, -90, 180]),
])
