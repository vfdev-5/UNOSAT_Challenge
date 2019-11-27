# [UNOSAT Challenge](https://challenge.phi-unet.com/)


## TODO

* [ ] EDA
    - VV / VH float images
    - ortho-rectified => zero-pixel zones
    - what are exactly the targets ?
    - quality of targets ?
    - same shape for multiple images of the same region -> still correct targets ?
    - shape rasterization produces mask for zero ortho-rectified part of image
    - on test data, we can reduce variance using multiple images per city

* [ ] Dataflow
    * [x] Merge VV / VH -> (VH, VV, (VH + VV) * 0.5)
    * [x] Rasterize shape masks
    * [x] Generate tiles from the data
    * [x] Generate tiles stats and fold indices
    * [x] Create pytorch datasets and dataloaders 

* [ ] Trainings
    * [x] Train a Light-Weight RefineNet model
    * [ ] Train DeeplabV3 model -> failed
    * [x] Sampling based on target

* [ ] Ideas to accelerate/improve training    
    * [ ] Implement data echoing/minibatch persistence to accelerate trainings ?
    * [ ] Label smoothing ?
    * [ ] Model with OctConv ?
    * [ ] Train a pre-segmentation classifier ? 
        => No need to do this. 
        => LB F1 score is computed on complete image which should have ground truth pixels
    * [x] Use CrossEntropy + Jaccard loss ?
    * [ ] Try to implement and train GSCNN model ?
    * [ ] Try Unsup Data Augmentation ?
    * [ ] Validation with TTA ?
        => depending on model, in some cases can improve predictions

* [ ] Inferences
    * [ ] Handler to save images with predictions: `[img, img+preds, preds, img+gt, gt]` with a metric value, e.g. `IoU(1)`    
    * [x] Handler to save predictions as images with geo info(tile level)
    * [x] Script to aggregate predictions as images for the same city
        * [x] Check if images has the same geo extension    
    * [x] Script to vectorize masks into shapefiles (city level)
    * [x] Add TTA using ttach package
    
* [x] Check submission score on which part of data => F1 score is computed on whole test dataset

## Results

Experiment | Validation IoU(1)| Validation F1 | Test F1 | Notes
---|---|---|---|---
[baseline_lwrefinenet.py](configs/train/baseline_lwrefinenet.py)| 0.648 | 0.891 | 0.688175 | LWRefineNet with CrossEntropy, validation city "38SNE"
[baseline_lwrefinenet_xentropy_jaccard.py](configs/train/baseline_lwrefinenet_xentropy_jaccard.py)| 0.668 | 0.899 |  | LWRefineNet with CrossEntropy+2*Jaccard, validation city "38SNE"
[baseline_lwrefinenet_xentropy_jaccard_tta.py](configs/train/baseline_lwrefinenet_xentropy_jaccard.py)| 0.668 | 0.899 |  | LWRefineNet with CrossEntropy+2*Jaccard, validation city "38SNE"


## Requirements

- mlflow
- git
- linux libs for opencv-python


## MLflow setup

Setup mlflow output path as 
```bash
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

## Data preparations

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


## Training, validation and inference

Training an a single node with single GPU:

```bash
export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/ --experiment-name=Trainings -P script_path=code/scripts/training.py -P config_path=configs/train/XXX.py
```

Validation:
```bash
export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/ --experiment-name=Inferences -P script_path=code/scripts/inference.py -P config_path=configs/inference/validate_XYZ.py
```

Inference on test data:
```bash
export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/ --experiment-name=Inferences -P script_path=code/scripts/inference.py -P config_path=configs/inference/test_XYZ.py
```


## Transform predictions to submission format

```bash
export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/ -e to_submission -P input_path=output/mlruns/2/XYZ/artifacts/raw
```

Shapefiles to submit are produced in the root of `input_path` folder.

### or manually every step

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

