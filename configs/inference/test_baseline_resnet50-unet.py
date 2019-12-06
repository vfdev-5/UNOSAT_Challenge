# Inference with baseline segmentation LWRefineNet
import os
from functools import partial

import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import Unet

import ttach as tta

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

batch_size = 3
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

model = Unet(encoder_name='resnet50', classes=2)
run_uuid = "48bddd1fcabb41acaedce035e16247cb"
weights_filename = "best_model_1_val_miou_bg=0.7517839.pth"


def custom_weights_loading(model, model_weights_filepath):
    state_dict = torch.load(model_weights_filepath)
    model.load_state_dict(state_dict['model'])


# TTA
tta_transforms = tta.Compose([
    tta.Rotate90(angles=[90, -90, 180]),
])

