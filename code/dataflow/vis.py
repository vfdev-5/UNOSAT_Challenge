from typing import Union, Callable, Optional, Sequence

import numpy as np
from PIL import Image

import torch

try:
    from image_dataset_viz import render_datapoint
except ImportError:
    raise RuntimeError("Install it via pip install --upgrade git+https://github.com/vfdev-5/ImageDatasetViz.git")


def _getvocpallete(num_cls):
    n = num_cls
    pallete = [0]*(n*3)
    for j in range(0, n):
        lab = j
        pallete[j*3+0] = 0
        pallete[j*3+1] = 0
        pallete[j*3+2] = 0
        i = 0
        while lab > 0:
            pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
            pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
            pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3
    return pallete


default_palette = _getvocpallete(256)


def render_mask(mask: Union[np.ndarray, Image.Image],
                palette: Sequence) -> Image.Image:
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    mask.putpalette(palette)
    mask = mask.convert(mode='RGB')
    return mask


def instance_normalize(x, axis=(0, 1)):
    vmin = np.percentile(x, q=2.5, axis=axis, keepdims=True)
    vmax = np.percentile(x, q=97.5, axis=axis, keepdims=True)
    x = np.clip(x, a_min=vmin, a_max=vmax)
    x = (x - vmin) / (vmax - vmin + 1e-10)
    return x


def render_image(img):
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    return (255 * instance_normalize(img)).astype(np.uint8)


from typing import Union, Callable, Optional, Sequence

import numpy as np
from PIL import Image

import torch

try:
    from image_dataset_viz import render_datapoint
except ImportError:
    raise RuntimeError("Install it via pip install --upgrade git+https://github.com/vfdev-5/ImageDatasetViz.git")


from dataflow.datasets import ISPRSTilesDataset


isprs_palette = []
for c in ISPRSTilesDataset.train_ids:
    isprs_palette += list(c)


def _getvocpallete(num_cls):
    n = num_cls
    pallete = [0]*(n*3)
    for j in range(0, n):
        lab = j
        pallete[j*3+0] = 0
        pallete[j*3+1] = 0
        pallete[j*3+2] = 0
        i = 0
        while lab > 0:
            pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
            pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
            pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3
    return pallete


default_palette = _getvocpallete(256)


def render_mask(mask: Union[np.ndarray, Image.Image],
                palette: Sequence) -> Image.Image:
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    mask.putpalette(palette)
    mask = mask.convert(mode='RGB')
    return mask


def tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
    img = t.cpu().numpy().transpose((1, 2, 0))
    return img.astype(np.uint8)


def make_grid(batch_img: torch.Tensor,
              batch_mask: torch.Tensor,
              img_denormalize_fn: Callable,
              mask_palette: Optional[Sequence] = default_palette,
              batch_gt_mask: Optional[torch.Tensor] = None):
    """Create a grid from batch image and mask as

        img1  | img2  | img3  | img4  | ...
        i+m1  | i+m2  | i+m3  | i+m4  | ...
        mask1 | mask2 | mask3 | mask4 | ...
        i+M1  | i+M2  | i+M3  | i+M4  | ...
        Mask1 | Mask2 | Mask3 | Mask4 | ...

        i+m = image + mask blended with alpha=0.4
        - maskN is predicted mask
        - MaskN is ground-truth mask if given

    Args:
        batch_img (torch.Tensor) batch of images of any type
        batch_mask (torch.Tensor) batch of masks
        img_denormalize_fn (Callable): function to denormalize batch of images
        mask_palette (list/tuple, optional): mask palette. Default Pascal VOC palette.
        batch_gt_mask (torch.Tensor, optional): batch of ground truth masks.
    """
    assert isinstance(batch_img, torch.Tensor) and isinstance(batch_mask, torch.Tensor)
    assert len(batch_img) == len(batch_mask)

    if batch_gt_mask is not None:
        assert isinstance(batch_gt_mask, torch.Tensor)
        assert len(batch_mask) == len(batch_gt_mask)

    b = batch_img.shape[0]
    h, w = batch_img.shape[2:]

    le = 3 if batch_gt_mask is None else 3 + 2
    out_image = np.zeros((h * le, w * b, 3), dtype='uint8')

    for i in range(b):
        img = batch_img[i]
        mask = batch_mask[i]

        img = img_denormalize_fn(img)
        img = tensor_to_rgb(img)
        mask = mask.cpu().numpy()
        mask = render_mask(mask, mask_palette)

        out_image[0:h, i * w:(i + 1) * w, :] = img
        out_image[1 * h:2 * h, i * w:(i + 1) * w, :] = render_datapoint(img,
                                                                        mask,
                                                                        blend_alpha=0.4)
        out_image[2 * h:3 * h, i * w:(i + 1) * w, :] = mask

        if batch_gt_mask is not None:
            gt_mask = batch_gt_mask[i]
            gt_mask = gt_mask.cpu().numpy()
            gt_mask = render_mask(gt_mask, mask_palette)
            out_image[3 * h:4 * h, i * w:(i + 1) * w, :] = render_datapoint(img,
                                                                            gt_mask,
                                                                            blend_alpha=0.4)
            out_image[4 * h:5 * h, i * w:(i + 1) * w, :] = gt_mask

    return out_image


def write_prediction(y_pred, filepath, palette=None):
    assert isinstance(y_pred, np.ndarray) and y_pred.ndim == 2, \
        "{} and {}".format(type(y_pred), y_pred.shape if isinstance(y_pred, np.ndarray) else None)

    filepath = filepath + ".png"
    pil_pred = Image.fromarray(y_pred)
    if palette is not None:
        pil_pred.putpalette(palette)
    pil_pred.save(filepath)


def write_prediction_on_image(y_pred, x, filepath, palette=default_palette):
    assert isinstance(y_pred, np.ndarray) and y_pred.ndim == 2, \
        "{} and {}".format(type(y_pred), y_pred.shape if isinstance(y_pred, np.ndarray) else None)

    assert isinstance(x, np.ndarray) and x.ndim == 3, \
        "{} and {}".format(type(x), x.shape if isinstance(x, np.ndarray) else None)

    pil_pred = Image.fromarray(y_pred)
    pil_pred.putpalette(palette)
    res = render_datapoint(x, pil_pred.convert('RGB'), blend_alpha=0.5)

    filepath = filepath + ".png"
    res.save(filepath)
