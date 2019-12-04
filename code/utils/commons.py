from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from ignite.engine import Events, Engine
from ignite.metrics import RunningAverage
from ignite.handlers import EarlyStopping, ModelCheckpoint, TerminateOnNan
from ignite.contrib.metrics import GpuInfo
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers import TensorboardLogger
import ignite.contrib.handlers.tensorboard_logger as tb_logger_module
from ignite.contrib.handlers import MLflowLogger
import ignite.contrib.handlers.mlflow_logger as mlflow_logger_module


def setup_trainer(train_update_function, model, optimizer, config, metric_names=None):

    trainer = Engine(train_update_function)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    lr_scheduler = config.lr_scheduler
    if isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
        trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda engine: lr_scheduler.step())
    else:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, config.lr_scheduler)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)

    # Checkpoint training
    checkpoint_handler = ModelCheckpoint(dirname=config.output_path.as_posix(),
                                         filename_prefix="checkpoint", 
                                         require_empty=False)
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1000),
                              checkpoint_handler,
                              {'model': model, 'optimizer': optimizer})

    # Logging training with TQDM
    def output_transform(x, name):
        return x[name]

    if metric_names is not None:
        for n in metric_names:
            RunningAverage(output_transform=partial(output_transform, name=n), epoch_bound=False).attach(trainer, n)

    GpuInfo().attach(trainer, name='gpu')
    ProgressBar(persist=False).attach(trainer, metric_names='all')
    ProgressBar(persist=True, bar_format="").attach(trainer,
                                                    event_name=Events.EPOCH_STARTED,
                                                    closing_event_name=Events.COMPLETED)
    return trainer


def setup_evaluators(model, device, val_metrics, config):

    prepare_batch = config.prepare_batch
    non_blocking = getattr(config, "non_blocking", True)
    model_output_transform = getattr(config, "model_output_transform", lambda x: x)

    def eval_update_function(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            y_pred = model_output_transform(y_pred)
            return {
                "y_pred": y_pred,
                "y": y
            }

    train_evaluator = Engine(eval_update_function)
    evaluator = Engine(eval_update_function)

    for name, metric in val_metrics.items():        
        metric.attach(train_evaluator, name)
        metric.attach(evaluator, name)

    ProgressBar(persist=False, desc="Train Evaluation").attach(train_evaluator)
    ProgressBar(persist=False, desc="Val Evaluation").attach(evaluator)

    train_evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)
    evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)

    return train_evaluator, evaluator


class dynamic:

    def __init__(self, obj, attr):
        self.obj = obj
        self.attr = attr

    def __repr__(self):
        return "{}".format(getattr(self.obj, self.attr))


def _setup_x_logging(logger, logger_module, trainer, optimizer, test_evaluator, dev_evaluator):
    logger.attach(trainer,
                  log_handler=logger_module.OutputHandler(tag="training", metric_names='all'),
                  event_name=Events.ITERATION_COMPLETED)

    # Log optimizer parameters
    logger.attach(trainer,
                  log_handler=logger_module.OptimizerParamsHandler(optimizer, param_name="lr"),
                  event_name=Events.ITERATION_STARTED)

    logger.attach(test_evaluator,
                  log_handler=logger_module.OutputHandler(tag=dynamic(test_evaluator, "tag"),
                                                          metric_names='all',
                                                          another_engine=trainer),
                  event_name=Events.COMPLETED)

    # Log val metrics:
    logger.attach(dev_evaluator,
                  log_handler=logger_module.OutputHandler(tag="validation",
                                                          metric_names='all',
                                                          another_engine=trainer),
                  event_name=Events.COMPLETED)


def setup_tb_logging(trainer, optimizer, train_evaluator, evaluator, config):
    tb_logger = TensorboardLogger(log_dir=config.output_path.as_posix())
    _setup_x_logging(tb_logger, tb_logger_module, trainer, optimizer, train_evaluator, evaluator)
    return tb_logger


def setup_mlflow_logging(trainer, optimizer, test_evaluator, dev_evaluator, **kwargs):
    mlflow_logger = MLflowLogger()
    _setup_x_logging(mlflow_logger, mlflow_logger_module, trainer, optimizer, test_evaluator, dev_evaluator)
    return mlflow_logger


def get_default_score_fn(metric_name):

    def wrapper(engine):
        score = engine.state.metrics[metric_name]
        return score

    return wrapper


def save_best_model_by_val_score(evaluator, model, metric_name, config):

    score_function = getattr(config, "score_function", get_default_score_fn(metric_name))

    best_model_handler = ModelCheckpoint(dirname=config.output_path.as_posix(),
                                         filename_prefix="best",
                                         n_saved=3,
                                         require_empty=False,
                                         score_name="val_{}".format(metric_name.lower().replace(" ", "_")),
                                         score_function=score_function)
    evaluator.add_event_handler(Events.COMPLETED, best_model_handler, {'model': model, })


def add_early_stopping_by_val_score(evaluator, trainer, metric_name, config):

    score_function = getattr(config, "score_function", get_default_score_fn(metric_name))
    es_handler = EarlyStopping(patience=config.es_patience, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, es_handler)


def empty_cuda_cache(engine):
    torch.cuda.empty_cache()
    import gc
    gc.collect()


def get_artifact_path(run_uuid, path):
    import mlflow
    client = mlflow.tracking.MlflowClient()
    return client.download_artifacts(run_uuid, path)
