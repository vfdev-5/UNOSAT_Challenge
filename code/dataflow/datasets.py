from typing import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, Subset

from dataflow.transforms import TransformedDataset
from dataflow.io_utils import read_image


class UnoSatTiles(Dataset):

    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Path '{path.as_posix()}' is not found")

        masks_path = path / "masks"
        if not masks_path.exists():
            raise ValueError(f"Path '{masks_path.as_posix()}' is not found")
        images_path = path / "images"
        if not images_path.exists():
            raise ValueError(f"Path '{images_path.as_posix()}' is not found")

        self.path = path
        self.masks = sorted(list(masks_path.rglob("*.tif")))
        if len(self.masks) < 1:
            raise RuntimeError(f"No mask images found at '{masks_path.as_posix()}'")

        self.images = [self._get_image_from_mask(m) for m in self.masks]

        for i in [int(c * len(self.images)) for c in [0.0, 0.1, 0.2, 0.5, 0.9]]:
            p = self.images[i]
            if not p.exists():
                raise RuntimeError(f"No images found at '{p.as_posix()}'")

    @staticmethod
    def _get_image_from_mask(mask):
        """Get image path from mask path:

        /path/masks/tile_id/tile_X_Y.tif
        """
        tile_id = mask.name
        tile_folder = mask.parent.name.replace("_vh_", "_3b_")
        return mask.parent.parent.parent / "images" / tile_folder / tile_id

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        return {
            "image": self.images[index],
            "mask": self.masks[index],
            "meta": {
                "image_path": self.images[index].as_posix(),
                "index": index
            }
        }


def read_img_mask(image, mask, **kwargs):
    kwargs['image'] = read_image(image.as_posix(), dtype='float32')
    kwargs['mask'] = read_image(mask, dtype='uint8')
    return kwargs


def linear_to_decibel(band):
    band[band < 1e-8] = 1.0
    return 10.0 * np.log10(band)


def read_img_in_db_with_mask(image, mask, **kwargs):
    img = read_image(image.as_posix(), dtype='float32')
    img = linear_to_decibel(img)
    kwargs['image'] = img
    kwargs['mask'] = read_image(mask, dtype='uint8')
    return kwargs


def read_img_5b_in_db_with_mask(image, mask, **kwargs):
    img = read_image(image.as_posix(), dtype='float32')
    img = linear_to_decibel(img)
    
    img5b = np.empty((img.shape[0], img.shape[1], 5), dtype=img.dtype)
    img5b[:, :, (0, 1, 2)] = img
    img5b[:, :, 3] = img[:, :, 1] * img[:, :, 0]
    img5b[:, :, 4] = (img[:, :, 1] + 1.0) / (img[:, :, 0] + 1.0)

    kwargs['image'] = img5b
    kwargs['mask'] = read_image(mask, dtype='uint8')
    return kwargs


def read_nimg_sqrt_mask(image, mask, **kwargs):
    img = read_image(image.as_posix(), dtype='float32')

    mins = img.reshape((-1, img.shape[2])).min(axis=0)[None, None, :]
    maxs = img.reshape((-1, img.shape[2])).max(axis=0)[None, None, :]
    nimg = (img - mins) / (maxs - mins + 1e-15)

    nimg = np.power(nimg, 0.2) - 0.5

    kwargs['image'] = nimg
    kwargs['mask'] = read_image(mask, dtype='uint8')
    return kwargs


def read_img_2log_with_mask(image, mask, **kwargs):
    img = read_image(image.as_posix(), dtype='float32')
    img[img == 0] = np.exp(-5.0)
    kwargs['image'] = np.log(img ** 2)
    kwargs['mask'] = read_image(mask, dtype='uint8')
    return kwargs


def read_img_as_6b_with_mask(image, mask, **kwargs):
    """Method to read image and mask, transform image to 5 channels:
    (3 original bands, vv * vh, (vv + 0.01) / (vh + 0.01), sqrt(b1^2 + b2^2))
    """
    img3b = read_image(image.as_posix(), dtype='float32')

    img6b = np.empty((img3b.shape[0], img3b.shape[1], 6), dtype=img3b.dtype)
    img6b[:, :, (0, 1, 2)] = img3b
    img6b[:, :, 3] = img3b[:, :, 1] * img3b[:, :, 0]
    img6b[:, :, 4] = (img3b[:, :, 1] + 0.01) / (img3b[:, :, 0] + 0.01)
    img6b[:, :, 5] = np.linalg.norm(img3b[:, :, (0, 1)], axis=-1)

    kwargs['image'] = img6b
    kwargs['mask'] = read_image(mask, dtype='uint8')
    return kwargs


def read_img_as_5b_with_mask(image, mask, **kwargs):
    """Method to read image and mask, transform image to 5 channels:
    (3 original bands, vv - vh, sqrt(b1^2 + b2^2))
    """
    img3b = read_image(image.as_posix(), dtype='float32')

    img5b = np.empty((img3b.shape[0], img3b.shape[1], 5), dtype=img3b.dtype)
    img5b[:, :, (0, 1, 2)] = img3b
    img5b[:, :, 3] = img3b[:, :, 1] - img3b[:, :, 0]
    img5b[:, :, 4] = np.linalg.norm(img3b[:, :, (0, 1)], axis=-1)

    kwargs['image'] = img5b
    kwargs['mask'] = read_image(mask, dtype='uint8')
    return kwargs


def read_img_only(image, **kwargs):
    kwargs['image'] = read_image(image.as_posix(), dtype='float32')
    return kwargs


def read_img_in_db(image, **kwargs):
    img = read_image(image.as_posix(), dtype='float32')
    img = linear_to_decibel(img)
    kwargs['image'] = img
    return kwargs


def get_fold_indices_dict(df, num_folds):
    folds_indices = {
        i: list(df[df['fold_index'] == i].index)
        for i in range(num_folds)
    }
    return folds_indices


def train_val_split(dataset: Dataset,
                    train_indices: Sequence,
                    val_indices: Sequence):
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def get_trainval_datasets(path, csv_path, train_folds, val_folds, read_img_mask_fn=read_img_mask):
    ds = UnoSatTiles(path)
    df = pd.read_csv(csv_path)
    # remove tiles to skip
    df = df[~df['skip']].copy()

    num_folds = len(df['fold_index'].unique())

    def _get_indices(fold_indices, folds_indices_dict):
        indices = []
        for i in fold_indices:
            indices += folds_indices_dict[i]
        return indices

    folds_indices_dict = get_fold_indices_dict(df, num_folds)
    train_indices = _get_indices(train_folds, folds_indices_dict)
    val_indices = _get_indices(val_folds, folds_indices_dict)
    train_ds, val_ds = train_val_split(ds, train_indices, val_indices)

    # Include data reading transformation
    train_ds = TransformedDataset(train_ds, transform_fn=read_img_mask_fn)
    val_ds = TransformedDataset(val_ds, transform_fn=read_img_mask_fn)

    return train_ds, val_ds


class UnoSatTestTiles(Dataset):

    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Path '{path.as_posix()}' is not found")

        images_path = path / "images"
        if not images_path.exists():
            raise ValueError(f"Path '{images_path.as_posix()}' is not found")

        self.path = path
        self.images = sorted(list(images_path.rglob("*.tif")))
        if len(self.images) < 1:
            raise RuntimeError(f"No images found at '{images_path.as_posix()}'")

        for i in [int(c * len(self.images)) for c in [0.0, 0.1, 0.2, 0.5, 0.9]]:
            p = self.images[i]
            if not p.exists():
                raise RuntimeError(f"No images found at '{p.as_posix()}'")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {
            "image": self.images[index],
            "meta": {
                "image_path": self.images[index].as_posix(),
                "index": index
            }
        }
