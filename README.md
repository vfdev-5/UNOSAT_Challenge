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
    * [ ] Train DeeplabV3 model
    * [x] Sampling based on target

* [ ] Ideas to accelerate/improve training    
    * [ ] Implement data echoing/minibatch persistence to accelerate trainings 
    * [ ] Label smoothing
    * [ ] Model with OctConv
    * [ ] Train a pre-segmentation classifier ?
    * [ ] Use CrossEntropy + Jaccard loss
    * [ ] Try to implement and train GSCNN model

* [ ] Training strategy 1
    * [ ] Train on all data without validation
        * [ ] 

* [ ] Inferences
    * [ ] Handler to save images with predictions: `[img, img+preds, preds, img+gt, gt]` with a metric value, e.g. `IoU(1)`    
    * [x] Handler to save predictions as images with geo info(tile level)
    * [x] Script to aggregate predictions as images for the same city
        * [x] Check if images has the same geo extension    
    * [x] Script to vectorize masks into shapefiles (city level)    
    
* [ ] Check submission score on which part of data


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

## Transform predictions to submission format

```bash
export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/ -e to_submission -P input_path=output/mlruns/2/XYZ/artifacts/raw
```

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

