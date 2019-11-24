from pathlib import Path

import torch

from dataflow.vis import make_grid, write_prediction, write_prediction_on_image, tensor_to_numpy, render_image
from dataflow.io_utils import write_prediction_with_geoinfo


def predictions_gt_images_handler(img_denormalize_fn, n_images=None, another_engine=None, prefix_tag=None):

    def wrapper(engine, logger, event_name):
        batch = engine.state.batch
        output = engine.state.output
        x = batch['image']
        y = batch['mask']
        y_pred = output['y_pred']

        if y.shape == y_pred.shape and y.ndim == 4:
            # Case of y of shape (B, C, H, W)
            y = torch.argmax(y, dim=1)

        y_pred = torch.argmax(y_pred, dim=1).byte()

        if n_images is not None:
            x = x[:n_images, ...]
            y = y[:n_images, ...]
            y_pred = y_pred[:n_images, ...]

        grid_pred_gt = make_grid(x, y_pred, img_denormalize_fn, batch_gt_mask=y)

        state = engine.state if another_engine is None else another_engine.state
        global_step = state.get_event_attrib_value(event_name)

        tag = "predictions_with_gt"
        if prefix_tag is not None:
            tag = "{}: {}".format(prefix_tag, tag)
        logger.writer.add_image(tag=tag, img_tensor=grid_pred_gt, global_step=global_step, dataformats='HWC')

    return wrapper


def _check_meta(output):
    if output['meta'] is None:
        raise RuntimeError("Output does not contain metadata info, available keys: {}"
                           .format(list(output.keys)))

    meta = output['meta']
    if not (isinstance(meta, dict) and "image_path" in meta and "index" in meta):
        raise RuntimeError("Output meta should be a dict and contain the keys: index, image_path; "
                           "but have {}".format(list(meta.keys())))


def _default_meta_image_path_transform(output_path, meta_image_path):
    output_path = Path(output_path)
    fname = Path(meta_image_path).stem
    subfolder = Path(meta_image_path).parent.name
    (output_path / subfolder).mkdir(exist_ok=True, parents=True)
    fp = output_path / subfolder / fname
    return fp.as_posix()


def save_raw_predictions(engine, output_path, palette=None, meta_image_path_transform=None):
    output = engine.state.output

    _check_meta(output)
    meta = output['meta']

    if meta_image_path_transform is None:
        meta_image_path_transform = _default_meta_image_path_transform

    # Save raw predictions
    y_probas = output['y_pred']
    y_preds = torch.argmax(y_probas, dim=1).byte()
    y_preds = y_preds.cpu().numpy()
    for y_pred, p in zip(y_preds, meta['image_path']):
        fp = meta_image_path_transform(output_path, p)
        write_prediction(y_pred, fp, palette=palette)


def save_overlayed_predictions(engine, output_path, img_denormalize_fn,
                               palette=None, meta_image_path_transform=None):
    batch = engine.state.batch
    output = engine.state.output

    _check_meta(output)
    meta = output['meta']

    if meta_image_path_transform is None:
        meta_image_path_transform = _default_meta_image_path_transform

    # Save overlayed predictions
    y_probas = output['y_pred']
    y_preds = torch.argmax(y_probas, dim=1).byte()
    y_preds = y_preds.cpu().numpy()

    imgs = batch['image']
    for x, y_pred, p in zip(imgs, y_preds, meta['image_path']):
        fp = meta_image_path_transform(output_path, p)
        x = img_denormalize_fn(x)
        x = tensor_to_numpy(x)
        x = render_image(x)

        write_prediction_on_image(y_pred, x, fp, palette=palette)


def save_raw_predictions_with_geoinfo(engine, output_path, meta_image_path_transform=None):
    output = engine.state.output

    _check_meta(output)
    meta = output['meta']

    if meta_image_path_transform is None:
        meta_image_path_transform = _default_meta_image_path_transform

    # Save raw predictions
    y_probas = output['y_pred']
    y_preds = torch.argmax(y_probas, dim=1).byte()
    y_preds = y_preds.cpu().numpy()
    for y_pred, p in zip(y_preds, meta['image_path']):
        fp = meta_image_path_transform(output_path, p)
        write_prediction_with_geoinfo(y_pred, fp, ref_image_fp=p)


def report_exception(engine, e):
    output = engine.state.output
    _check_meta(output)
    meta = output['meta']
    print("Exception {} raised on data: {}".format(e, meta['image_path']))
    raise RuntimeError(e)
