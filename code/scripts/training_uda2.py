# This a training script launched with py_config_runner
# It should obligatory contain `run(config, **kwargs)` method

# Training based on Unsupervised Data Augmentation training: https://arxiv.org/abs/1904.12848

from collections.abc import Mapping
from pathlib import Path

import random

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from apex import amp
from apex.parallel import DistributedDataParallel as DDP

import mlflow

import ignite
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.utils import convert_tensor
from ignite.metrics import Fbeta, IoU, ConfusionMatrix, mIoU
from ignite.metrics.confusion_matrix import cmAccuracy, cmPrecision, cmRecall
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.engines import common

from py_config_runner.config_utils import get_params, TRAINVAL_CONFIG, assert_config
from py_config_runner.utils import set_seed

from utils.handlers import predictions_gt_images_handler


TRAINVAL_CONFIG += (
    ('unsup_criterion', torch.nn.Module),
    ('unsup_train_loader', DataLoader)
)


def training(config, local_rank, with_pbar_on_iters=True):

    if not getattr(config, "use_fp16", True):
        raise RuntimeError("This training script uses by default fp16 AMP")

    set_seed(config.seed + local_rank)
    torch.cuda.set_device(local_rank)
    device = 'cuda'

    torch.backends.cudnn.benchmark = True

    train_loader = config.train_loader
    train_sampler = getattr(train_loader, "sampler", None)
    assert train_sampler is not None, "Train loader of type '{}' " \
                                      "should have attribute 'sampler'".format(type(train_loader))
    assert hasattr(train_sampler, 'set_epoch') and callable(train_sampler.set_epoch), \
        "Train sampler should have a callable method `set_epoch`"

    unsup_train_loader = config.unsup_train_loader
    unsup_train_sampler = getattr(unsup_train_loader, "sampler", None)
    assert unsup_train_sampler is not None, "Train loader of type '{}' " \
                                      "should have attribute 'sampler'".format(type(unsup_train_loader))
    assert hasattr(unsup_train_sampler, 'set_epoch') and callable(unsup_train_sampler.set_epoch), \
        "Unsupervised train sampler should have a callable method `set_epoch`"

    train_eval_loader = config.train_eval_loader
    val_loader = config.val_loader

    model = config.model.to(device)
    optimizer = config.optimizer
    model, optimizer = amp.initialize(model, optimizer, opt_level=getattr(config, "fp16_opt_level", "O2"), num_losses=2)
    model = DDP(model, delay_allreduce=True)
    
    criterion = config.criterion.to(device)
    unsup_criterion = config.unsup_criterion.to(device)
    unsup_batch_num_repetitions = getattr(config, "unsup_batch_num_repetitions", 1)

    # Setup trainer
    prepare_batch = getattr(config, "prepare_batch")
    non_blocking = getattr(config, "non_blocking", True)
    accumulation_steps = getattr(config, "accumulation_steps", 1)
    model_output_transform = getattr(config, "model_output_transform", lambda x: x)

    def cycle(seq):
        while True:
            for i in seq:
                yield i

    unsup_train_loader_iter = cycle(unsup_train_loader)

    def supervised_loss(batch):
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        y_pred = model_output_transform(y_pred)
        loss = criterion(y_pred, y)
        return loss

    def unsupervised_loss(x):

        with torch.no_grad():
            y_pred_orig = model(x)

            # Data augmentation: geom only
            k = random.randint(1, 3)
            x_aug = torch.rot90(x, k=k, dims=(2, 3))
            y_pred_orig_aug = torch.rot90(y_pred_orig, k=k, dims=(2, 3))
            if random.random() < 0.5:
                x_aug = torch.flip(x_aug, dims=(2, ))
                y_pred_orig_aug = torch.flip(y_pred_orig_aug, dims=(2, )) 
            if random.random() < 0.5:
                x_aug = torch.flip(x_aug, dims=(3, ))
                y_pred_orig_aug = torch.flip(y_pred_orig_aug, dims=(3, )) 

            y_pred_orig_aug = y_pred_orig_aug.argmax(dim=1).long()

        y_pred_aug = model(x_aug.detach())

        loss = unsup_criterion(y_pred_aug, y_pred_orig_aug.detach())

        return loss

    def train_update_function(engine, batch):
        model.train()

        loss = supervised_loss(batch)
        if isinstance(loss, Mapping):
            assert 'supervised batch loss' in loss
            loss_dict = loss
            output = {k: v.item() for k, v in loss_dict.items()}
            loss = loss_dict['supervised batch loss'] / accumulation_steps
        else:
            output = {'supervised batch loss': loss.item()}
        
        # Difference with original UDA
        # Apply separately grads from supervised/unsupervised parts
        with amp.scale_loss(loss, optimizer, loss_id=0) as scaled_loss:
            scaled_loss.backward()

        if engine.state.iteration % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        unsup_batch = next(unsup_train_loader_iter)
        unsup_x = unsup_batch['image']
        unsup_x = convert_tensor(unsup_x, device=device, non_blocking=non_blocking)

        for _ in range(unsup_batch_num_repetitions):
            unsup_loss = engine.state.unsup_lambda * unsupervised_loss(unsup_x)

            assert isinstance(unsup_loss, torch.Tensor)
            output['unsupervised batch loss'] = unsup_loss.item()

            with amp.scale_loss(unsup_loss, optimizer, loss_id=1) as scaled_loss:
                scaled_loss.backward()

            if engine.state.iteration % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        unsup_batch = None
        unsup_x = None

        total_loss = loss +  unsup_loss
        output['total batch loss'] = total_loss.item()

        return output

    output_names = getattr(config, "output_names", 
                           ['supervised batch loss', 'unsupervised batch loss', 'total batch loss'])

    trainer = Engine(train_update_function)

    @trainer.on(Events.STARTED)
    def init(engine):
        if hasattr(config, "unsup_lambda_min"):
            engine.state.unsup_lambda = config.unsup_lambda_min
        else:
            engine.state.unsup_lambda = getattr(config, "unsup_lambda", 0.001)

    @trainer.on(Events.ITERATION_COMPLETED)
    def update_unsup_params(engine):        
        engine.state.unsup_lambda += getattr(config, "unsup_lambda_delta", 0.00001)
        if hasattr(config, "unsup_lambda_max"):
            m = config.unsup_lambda_max
            engine.state.unsup_lambda = engine.state.unsup_lambda if engine.state.unsup_lambda < m else m

    common.setup_common_distrib_training_handlers(
        trainer, train_sampler,
        to_save={'model': model, 'optimizer': optimizer},
        save_every_iters=1000, output_path=config.output_path.as_posix(),
        lr_scheduler=config.lr_scheduler, output_names=output_names,
        with_pbars=True, with_pbar_on_iters=with_pbar_on_iters, log_every_iters=1
    )

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

    if hasattr(config, "val_metrics") and isinstance(config.val_metrics, dict):
        val_metrics.update(config.val_metrics)

    evaluator_args = dict(
        model=model, metrics=val_metrics, device=device, non_blocking=non_blocking, prepare_batch=prepare_batch,
        output_transform=lambda x, y, y_pred: {'y_pred': model_output_transform(y_pred), 'y': y}
    )
    train_evaluator = create_supervised_evaluator(**evaluator_args)
    evaluator = create_supervised_evaluator(**evaluator_args)

    if dist.get_rank() == 0 and with_pbar_on_iters:
        ProgressBar(persist=False, desc="Train Evaluation").attach(train_evaluator)
        ProgressBar(persist=False, desc="Val Evaluation").attach(evaluator)

    def run_validation(engine):
        train_evaluator.run(train_eval_loader)
        evaluator.run(val_loader)

    if getattr(config, "start_by_validation", False):
        trainer.add_event_handler(Events.STARTED, run_validation)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=getattr(config, "val_interval", 1)), run_validation)
    trainer.add_event_handler(Events.COMPLETED, run_validation)

    score_metric_name = "mIoU_bg"

    if hasattr(config, "es_patience"):
        common.add_early_stopping_by_val_score(config.es_patience, evaluator, trainer, metric_name=score_metric_name)

    if dist.get_rank() == 0:

        tb_logger = common.setup_tb_logging(config.output_path.as_posix(), trainer, optimizer,
                                            evaluators={"training": train_evaluator, "validation": evaluator})
        common.setup_mlflow_logging(trainer, optimizer,
                                    evaluators={"training": train_evaluator, "validation": evaluator})

        common.save_best_model_by_val_score(config.output_path.as_posix(), evaluator, model,
                                            metric_name=score_metric_name, trainer=trainer)

        # Log unsup_lambda
        @trainer.on(Events.ITERATION_COMPLETED(every=100))
        def tblog_unsupervised_lambda(engine):
            tb_logger.writer.add_scalar("training/unsupervised lambda", engine.state.unsup_lambda, engine.state.iteration)
            mlflow.log_metric("training unsupervised lambda", engine.state.unsup_lambda, step=engine.state.iteration)

        # Log train/val predictions:
        tb_logger.attach(evaluator,
                         log_handler=predictions_gt_images_handler(img_denormalize_fn=config.img_denormalize,
                                                                   n_images=15,
                                                                   another_engine=trainer,
                                                                   prefix_tag="validation"),
                         event_name=Events.ITERATION_COMPLETED(once=len(val_loader) // 2))

        log_train_predictions = getattr(config, "log_train_predictions", False)
        if log_train_predictions:
            tb_logger.attach(train_evaluator,
                             log_handler=predictions_gt_images_handler(img_denormalize_fn=config.img_denormalize,
                                                                       n_images=15,
                                                                       another_engine=trainer,
                                                                       prefix_tag="validation"),
                             event_name=Events.ITERATION_COMPLETED(once=len(train_eval_loader) // 2))

    trainer.run(train_loader, max_epochs=config.num_epochs)


def run(config, logger=None, local_rank=0, **kwargs):

    assert torch.cuda.is_available()
    assert torch.backends.cudnn.enabled, "Nvidia/Amp requires cudnn backend to be enabled."

    dist.init_process_group("nccl", init_method="env://")

    # As we passed config with option --manual_config_load
    assert hasattr(config, "setup"), "We need to manually setup the configuration, please set --manual_config_load " \
                                     "to py_config_runner"

    config = config.setup()

    assert_config(config, TRAINVAL_CONFIG)
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
        mlflow.log_params(get_params(config, TRAINVAL_CONFIG))
        mlflow.log_params({'mean': config.mean, 'std': config.std})

    try:
        import os

        with_pbar_on_iters = True
        if "DISABLE_PBAR_ON_ITERS" in os.environ:
            with_pbar_on_iters = False

        training(config, local_rank=local_rank, with_pbar_on_iters=with_pbar_on_iters)
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
