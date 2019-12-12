# Script to validate predictions and compute metrics

import argparse

from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

import tqdm

import mlflow

from joblib import Parallel, delayed

from dataflow.io_utils import read_image


def worker_task(pred_file, gt_path):
    fname = pred_file.name
    parent = pred_file.parent.name
    gt_file = gt_path / parent / fname

    if not gt_file.exists():
        gt_file = gt_path / (parent.replace("_3b_", "_vh_")) / fname

    assert gt_file.exists(), "File is not found {}".format(gt_file.as_posix())
    
    y_pred = read_image(pred_file, dtype='int')
    y_true = read_image(gt_file, dtype='int')    
    
    return confusion_matrix(y_true.ravel(), y_pred.ravel(), labels=[0, 1])


def run(preds_path, gt_path):        
    
    pred_files = sorted(list(preds_path.rglob("*.tif")))

    with Parallel(n_jobs=10) as parallel:
        res = parallel(delayed(worker_task)(fp, gt_path) for fp in tqdm.tqdm(pred_files))

    cm = np.array(res)
    cm = cm.sum(axis=0)
    iou = cm.diagonal() / (cm.sum(axis=1) + cm.sum(axis=0) - cm.diagonal() + 1e-15)
    miou = iou.mean()
    pr = cm.diagonal() / (cm.sum(axis=0) + 1e-15)
    re = cm.diagonal() / (cm.sum(axis=1) + 1e-15)
    f1 = ((2.0) * pr * re / (pr + re + 1e-15)).mean()

    mlflow.log_metrics({
        "validation F1": f1, 
        "validation mIoU_bg": miou,
        "validation IoU 1": iou[1],
    })


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Validate predictions")
    parser.add_argument("preds_path", type=str, help="Path to predictions")
    parser.add_argument("gt_path", type=str, help="Path to ground-truth")

    args = parser.parse_args()
    
    preds_path = Path(args.preds_path)    
    assert preds_path.exists(), f"Not found {preds_path.as_posix()}"
    
    gt_path = Path(args.gt_path)    
    assert gt_path.exists(), f"Not found {gt_path.as_posix()}"

    run(preds_path, gt_path)
