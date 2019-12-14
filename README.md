# [UNOSAT Challenge](https://challenge.phi-unet.com/)

Humanitarian AI4EO Challenge For UNOSAT, ESA and CERN openlab to detect building footprints in Iraq and support the local government to plan reconstruction and development activities in the area.

## Results for Phase 1

We are training conv neural networks for designed for the segmentation task:

Experiment | Validation IoU(1) | Validation F1 | Test F1 | Notes
---|---|---|---|---
[baseline_lwrefinenet.py](configs/train/baseline_lwrefinenet.py)| 0.506 | 0.83 | 0.688175 | LWRefineNet with CrossEntropy, validation city "38SNE", c01b6ccf59474808b528f55ee13b497d
[baseline_lwrefinenet_xentropy_jaccard.py](configs/train/baseline_lwrefinenet_xentropy_jaccard.py)| 0.524 | 0.838 | 0.705516 | LWRefineNet with CrossEntropy+2*Jaccard, validation city "38SNE", inference with TTA, df7c6ed4870a40c1b9dcf742a6c07f0a
[baseline_resnet50-unet.py](configs/train/baseline_resnet50-unet.py)| 0.525 | 0.839 | 0.701196 | ResNet50+UNet with CrossEntropy, validation city "38SNE", inference with TTA, 2982bbc723e049f4839ea83d1900c2a2
[baseline_lwrefinenet_on_5b_db.py](configs/train/baseline_lwrefinenet_on_5b_db.py)| 0.532 | 0.841 | - | LWRefineNet with CrossEntropy on 5 bands (dB), validation city "38SNE", inference with TTA, 48b27adb07794d6c901047307d304312
[baseline_se_resnext50-FPN_on_db.py](configs/train/baseline_se_resnext50-FPN_on_db.py)| 0.535 | 0.843 | 0.742955 | SE-ResNet50+FPN with CrossEntropy on 3 bands (dB), validation city "38SNE", inference with TTA, 38c8cca75f8b46a798224a146cdf4426
[baseline_se_resnext50-FPN_on_db_lr_restart:py](configs/train/baseline_se_resnext50-FPN_on_db_lr_restart.py)| 0.543 | 0.847 | 0.643489 | SE-ResNet50+FPN with CrossEntropy, other hyperparams, LR restarts, validation city "38SNE", inference with TTA, 3a0b1378668547f0967974fdb56bb710


## Code architecture

- code : project package providing data processing, training/validation/inference scripts and modules with dataflow, losses, models and utils.
    - [code/scripts/training.py](code/scripts/training.py) training script (single node, 1/N GPUs).
    - [code/scripts/training_uda.py](code/scripts/training_uda.py) (Work-In-Progress) training script with Unsupervised Data Augmentation method (single node, 1/N GPUs).
    - [code/scripts/inference.py](code/scripts/inference.py) inference script to validate a model or make predictions on the test dataset (single node, 1/N GPUs).
- configs : configuration files
    - configuration is a python file, flexible and highly configurable without any meta-language
- experiments : some bash scripts and [mlflow](https://mlflow.org/) related file to manage ML experiments in reproducible manner.
    - software dependencies are setup in [experiments/conda.yaml](experiments/conda.yaml)
    - job commands are defined in [experiments/MLproject](experiments/MLproject)
- notebooks : jupyter notebook for visual checkings and development.


### Requirements

- Linux OS, Python 3.X, pip
- git
- linux libs for opencv-python
- conda
- [mlflow](https://mlflow.org/) : `pip install mlflow`

### MLflow setup

Setup mlflow output path as 
```bash
cd UNOSAT_Challenge
export MLFLOW_TRACKING_URI=$PWD/output/mlruns
```

Create once "Trainings" and "Inferences" experiments
```bash
mlflow experiments create -n Trainings
mlflow experiments create -n Inferences
```
or check existing experiments:
```bash
mlflow experiments list
```

### Data setup and preparations

Create symbolic links to downloaded data and output folder
```bash
cd UNOSAT_Challenge
ln -s /path/to/data input
ln -s /path/to/output output
```

Setup mlflow tracking path
```bash
export MLFLOW_TRACKING_URI=$PWD/output/mlruns
```

Generate 3-bands files from VV / VH separate files
```bash
mlflow run experiments/ -e generate_3b_images -P input_path=../input/train -P output_path=../output/train/images_3b
mlflow run experiments/ -e generate_3b_images -P input_path=../input/test -P output_path=../output/test/images_3b
```

Rasterize shape files:
```bash
mlflow run experiments/ -e rasterize -P input_path=../input/train -P output_path=../output/train/masks
```

Generate train tiles:
```bash
mlflow run experiments/ -e generate_tiles -P input_path=../output/train/images_3b -P output_path=../input/train_tiles/images
mlflow run experiments/ -e generate_tiles -P input_path=../output/train/masks -P output_path=../input/train_tiles/masks
```

Generate test tiles:
```bash
mlflow run experiments/ -e generate_test_tiles -P input_path=../output/test/images_3b -P output_path=../input/test_tiles/images
```

Generate train tiles stats:
```bash
mlflow run experiments/ -e generate_tiles_stats -P input_path=../input/train_tiles/ -P output_path=../input/train_tiles/
```


### Training, validation and inference

Training an a single node with 1 or N GPUs:

```bash
export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/ --experiment-name=Trainings -P script_path=code/scripts/training.py -P config_path=configs/train/XXX.py -P num_gpus=1
```

Validation (load model, make prediction, compute metrics) on validation data:
```bash
export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/ --experiment-name=Inferences -P script_path=code/scripts/inference.py -P config_path=configs/inference/validate_XYZ.py
```

Inference on test data:
```bash
export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/ --experiment-name=Inferences -P script_path=code/scripts/inference.py -P config_path=configs/inference/test_XYZ.py
```

Ensembling multiple predictions 
```bash
export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/ --experiment-name=Inferences -e ensemble -P input_paths="$PWD/output/mlruns/2/48b27adb07794d6c901047307d304312/artifacts/raw/;$PWD/output/mlruns/2/38c8cca75f8b46a798224a146cdf4426/artifacts/raw/;$PWD/output/mlruns/2/3a0b1378668547f0967974fdb56bb710/artifacts/raw"
```

Run validation on predictions (validation tiles):
```bash
mlflow run experiments/ --experiment-name=Inferences -e validate -P preds_path=$PWD/output/mlruns/2/XYZ/artifacts/raw/ -P gt_path=$PWD/input/train_tiles/masks
```

### Transform predictions to submission format

```bash
export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/ -e to_submission -P input_path=output/mlruns/2/XYZ/artifacts/raw
```

Shapefiles to submit are produced in the root of `input_path` folder.

#### or manually every step

1) Merge tiles into a single mask image:
```bash
export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/ -e merge_tiles -P input_path=output/mlruns/2/XYZ/artifacts/raw
```

2) Aggregate predictions by city
```bash
mlflow run experiments/ -e all_agg_by_city -P input_path=output/mlruns/2/XYZ/artifacts/raw/
```

3) Vectorize predictions
```bash
mlflow run experiments/ -e polygonize -P input_path=output/mlruns/2/XYZ/artifacts/raw/
```

### MLflow dashboard

To visualize experiments and runs, user can start mlflow dashboard:

```bash
mlflow server --backend-store-uri $PWD/output/mlruns --default-artifact-root $PWD/output/mlruns -p 6026 -h 0.0.0.0
```

### Remove deleted MLflow runs 

```bash
for i in `mlflow runs list --experiment-id=1 -v deleted_only | awk '{ print $4 }' | awk '/[0-9]+/'`; do rm -R output/mlruns/1/$i; done
```

### TODO/Ideas

* [x] EDA
    - VV / VH float images
    - ortho-rectified => zero-pixel zones
    - what are exactly the targets ?
    - quality of targets ?
    - same shape for multiple images of the same region -> still correct targets ?
    - shape rasterization produces mask for zero ortho-rectified part of image
    - on test data, we can reduce variance using multiple images per city

* [x] Dataflow
    * [x] Merge VV / VH -> (VH, VV, (VH + VV) * 0.5)
    * [x] Rasterize shape masks
    * [x] Generate tiles from the data
    * [x] Generate tiles stats and fold indices
    * [x] Create pytorch datasets and dataloaders 

* [x] Baselines
    * [x] Train Light-Weight RefineNet model
    * [x] Train ResNet-101 DeeplabV3 model
    * [x] Train SE-ResNet-50-FPN model 
    * [x] Sampling based on target    

* [ ] Ideas to accelerate/improve training    
    * [ ] Implement data echoing/minibatch persistence to accelerate trainings ?
    * [ ] Label smoothing ?
    * [ ] Model with OctConv ?
    * [x] Train a pre-segmentation classifier ? 
        => No need to do this. 
        => LB F1 score is computed on complete image which should have ground truth pixels
    * [x] Use CrossEntropy + Jaccard loss ?
    * [ ] Try to implement and train GSCNN model ?
    * [ ] Try Unsup Data Augmentation ?
    * [ ] Try Pseudo-Labelling of test data ?
    * [x] Try to train on all train dataset and validate on test dataset
    * [ ] Try multiple-input architectures
    * [x] Validation with TTA ?
        => depending on model, in some cases can improve predictions
    * [x] Data normalization over training mean/std
        * [x] Validate with LWRefineNet model on 3b => same as original 3b
    * [x] Try different input data:
        * [x] Generate on fly 3 input channels transformed by log(x^2)
            * [x] Validate the change with LWRefineNet model => worse than original 3 channels
        * [x] Generate on fly 5 input channels: (VH, VV, (VH + VV) * 0.5, VH - VV, sqrt(VH^2 + VV^2))
            * [x] Validate the change with LWRefineNet model => same as original 3 channels
        * [x] Sample-wise min/max normalization with `x^0.2 - 0.5`
            * [x] Validate the change with LWRefineNet model => same or worse than original 3 channels
    * [x] More features and channel's normalization
        * [x] VV * VH, VV / VH => small improvements for LWRefineNet, did not work for SE-ResNet50-FPN
    

* [ ] Inferences
    * [ ] Handler to save images with predictions: `[img, img+preds, preds, img+gt, gt]` with a metric value, e.g. `IoU(1)`    
    * [x] Handler to save predictions as images with geo info(tile level)
    * [x] Script to aggregate predictions as images for the same city
        * [x] Check if images has the same geo extension    
    * [x] Script to vectorize masks into shapefiles (city level)
    * [x] Add TTA using ttach package
    
* [x] Check submission score on which part of data => F1 score is computed on whole test dataset

