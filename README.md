# [UNOSAT Challenge](https://challenge.phi-unet.com/)


## TODO

* [ ] EDA
    - VV / VH float images
    - ortho-rectified => zero-pixel zones
    - what are exactly the targets ?
    - quality of targets ?
    - same shape for multiple images of the same region -> still correct targets ?
    - shape rasterization produces mask for zero ortho-rectified part of image

* [ ] Dataflow
    * [x] Merge VV / VH -> (VH, VV, (VH + VV) * 0.5)
    * [x] Rasterize shape masks
    * [x] Generate tiles from the data
    * [x] Generate tiles stats and fold indices
    * [x] Create pytorch datasets and dataloaders 

* [ ] Trainings
    * [x] Train a Light-Weight RefineNet model
    * [x] Sampling based on target
    * [ ] Label smoothing
    * [ ] Train a pre-segmentation classifier ?
    * [ ] Try to implement and train GSCNN model

* [ ] Inferences
    * [ ] Handler to save images with predictions: `[img, img+preds, preds, img+gt, gt]` with a metric value, e.g. `IoU(1)`
    * [ ] Handler to vectorize masks   



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


## Training and inference

On a single node with single GPU:

```bash
export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/ --experiment-name=Trainings -P script_path=code/scripts/training.py -P config_path=configs/train/XXX.py
```

```bash
export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/ --experiment-name=Inferences -P script_path=code/scripts/inference.py -P config_path=configs/inference/XXX.py
```

### MLflow dashboard

To visualize experiments and runs, user can start mlflow dashboard:

```bash
mlflow server --backend-store-uri $PWD/output/mlruns --default-artifact-root $PWD/output/mlruns -p 6026 -h 0.0.0.0
```

