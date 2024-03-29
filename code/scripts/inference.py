# This a script to run inference, it is launched with py_config_runner
# It should obligatory contain `run(config, **kwargs)` method

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import mlflow

import ignite
from ignite.engine import Engine, Events
from ignite.metrics import Fbeta, IoU, ConfusionMatrix, mIoU
from ignite.metrics.confusion_matrix import cmAccuracy, cmPrecision, cmRecall
from ignite.contrib.handlers import MLflowLogger, ProgressBar
from ignite.contrib.handlers.mlflow_logger import OutputHandler

from py_config_runner.config_utils import get_params, TORCH_DL_BASE_CONFIG, assert_config
from py_config_runner.utils import set_seed

from utils.commons import get_artifact_path
from utils.handlers import save_raw_predictions_with_geoinfo, save_overlayed_predictions, report_exception
from dataflow.vis import default_palette


INFERENCE_CONFIG = TORCH_DL_BASE_CONFIG + (
    ("data_loader", DataLoader),
    ("model", torch.nn.Module),
    ("run_uuid", str),
    ("weights_filename", str)
)


def inference(config, local_rank, with_pbar_on_iters=True):

    set_seed(config.seed + local_rank)
    torch.cuda.set_device(local_rank)
    device = 'cuda'

    torch.backends.cudnn.benchmark = True

    # Load model and weights
    model_weights_filepath = Path(get_artifact_path(config.run_uuid, config.weights_filename))
    assert model_weights_filepath.exists(), \
        "Model weights file '{}' is not found".format(model_weights_filepath.as_posix())

    model = config.model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    if hasattr(config, "custom_weights_loading"):
        config.custom_weights_loading(model, model_weights_filepath)
    else:
        state_dict = torch.load(model_weights_filepath)
        if not all([k.startswith("module.") for k in state_dict]):
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    model.eval()

    prepare_batch = config.prepare_batch
    non_blocking = getattr(config, "non_blocking", True)
    model_output_transform = getattr(config, "model_output_transform", lambda x: x)

    tta_transforms = getattr(config, "tta_transforms", None)

    def eval_update_function(engine, batch):
        with torch.no_grad():
            x, y, meta = prepare_batch(batch, device=device, non_blocking=non_blocking)

            if tta_transforms is not None:
                y_preds = []
                for t in tta_transforms:
                    t_x = t.augment_image(x)
                    t_y_pred = model(t_x)
                    t_y_pred = model_output_transform(t_y_pred)
                    y_pred = t.deaugment_mask(t_y_pred)
                    y_preds.append(y_pred)

                y_preds = torch.stack(y_preds, dim=0)
                y_pred = torch.mean(y_preds, dim=0)
            else:
                y_pred = model(x)
                y_pred = model_output_transform(y_pred)
            return {
                "y_pred": y_pred,
                "y": y,
                "meta": meta
            }

    evaluator = Engine(eval_update_function)

    has_targets = getattr(config, "has_targets", False)

    if has_targets:
        def output_transform(output):
            return output['y_pred'], output['y']

        num_classes = config.num_classes
        cm_metric = ConfusionMatrix(num_classes=num_classes, output_transform=output_transform)
        pr = cmPrecision(cm_metric, average=False)
        re = cmRecall(cm_metric, average=False)

        val_metrics = {
            "IoU": IoU(cm_metric),
            "mIoU_bg": mIoU(cm_metric),
            "Accuracy": cmAccuracy(cm_metric),
            "Precision": pr,
            "Recall": re,
            "F1": Fbeta(beta=1.0, output_transform=output_transform)
        }

        if hasattr(config, "metrics") and isinstance(config.metrics, dict):
            val_metrics.update(config.metrics)

        for name, metric in val_metrics.items():
            metric.attach(evaluator, name)

        if dist.get_rank() == 0:
            # Log val metrics:
            mlflow_logger = MLflowLogger()
            mlflow_logger.attach(evaluator,
                                 log_handler=OutputHandler(tag="validation",
                                                           metric_names=list(val_metrics.keys())),
                                 event_name=Events.EPOCH_COMPLETED)

    if dist.get_rank() == 0 and with_pbar_on_iters:
        ProgressBar(persist=True, desc="Inference").attach(evaluator)

    if dist.get_rank() == 0:
        do_save_raw_predictions = getattr(config, "do_save_raw_predictions", True)
        do_save_overlayed_predictions = getattr(config, "do_save_overlayed_predictions", True)

        if not has_targets:
            assert do_save_raw_predictions or do_save_overlayed_predictions, \
                "If no targets, either do_save_overlayed_predictions or do_save_raw_predictions should be " \
                "defined in the config and has value equal True"

        # Save predictions
        if do_save_raw_predictions:
            raw_preds_path = config.output_path / "raw"
            raw_preds_path.mkdir(parents=True)

            evaluator.add_event_handler(Events.ITERATION_COMPLETED,
                                        save_raw_predictions_with_geoinfo,
                                        raw_preds_path)

        if do_save_overlayed_predictions:
            overlayed_preds_path = config.output_path / "overlay"
            overlayed_preds_path.mkdir(parents=True)

            evaluator.add_event_handler(Events.ITERATION_COMPLETED,
                                        save_overlayed_predictions,
                                        overlayed_preds_path,
                                        img_denormalize_fn=config.img_denormalize,
                                        palette=default_palette)

    evaluator.add_event_handler(Events.EXCEPTION_RAISED, report_exception)

    # Run evaluation
    evaluator.run(config.data_loader)


def run(config, logger=None, local_rank=0, **kwargs):

    assert torch.cuda.is_available()
    assert torch.backends.cudnn.enabled, "Nvidia/Amp requires cudnn backend to be enabled."

    dist.init_process_group("nccl", init_method="env://")

    # As we passed config with option --manual_config_load
    assert hasattr(config, "setup"), "We need to manually setup the configuration, please set --manual_config_load " \
                                     "to py_config_runner"

    config = config.setup()

    assert_config(config, INFERENCE_CONFIG)

    # The following attributes are automatically added by py_config_runner
    assert hasattr(config, "config_filepath") and isinstance(config.config_filepath, Path)
    assert hasattr(config, "script_filepath") and isinstance(config.script_filepath, Path)

    output_path = mlflow.get_artifact_uri()
    config.output_path = Path(output_path)

    if dist.get_rank() == 0:

        # dump python files to reproduce the run
        mlflow.log_artifact(config.config_filepath.as_posix())
        mlflow.log_artifact(config.script_filepath.as_posix())

        mlflow.log_params({
            "pytorch version": torch.__version__,
            "ignite version": ignite.__version__,
        })
        mlflow.log_params(get_params(config, INFERENCE_CONFIG))
        mlflow.log_params({'mean': config.mean, 'std': config.std})

    try:
        import os

        with_pbar_on_iters = True
        if "DISABLE_PBAR_ON_ITERS" in os.environ:
            with_pbar_on_iters = False

        inference(config, local_rank=local_rank, with_pbar_on_iters=with_pbar_on_iters)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if dist.get_rank() == 0:
            mlflow.log_param("Run Status", "FAILED")
        dist.destroy_process_group()
        raise e

    if dist.get_rank() == 0:
        mlflow.log_param("Run Status", "OK")
    dist.destroy_process_group()
