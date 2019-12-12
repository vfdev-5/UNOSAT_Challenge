# Script to ensemble predictions

import argparse

from pathlib import Path

import numpy as np

import tqdm

import rasterio as rio

import mlflow

from dataflow.io_utils import read_image


def aggregate_and_save(folders_to_agg, output_path):
    
    master_tiles = sorted(list([p.name for p in folders_to_agg[0].iterdir() if not p.is_dir()]))
    
    if not output_path.exists():
        output_path.mkdir(parents=True)

    for tile_name in master_tiles:
        # Skip merged files generated when `to_submission` job was exectuted on predictions
        if "merged" in tile_name:
            continue        
        ref_fp = folders_to_agg[0] / tile_name
        assert ref_fp.exists(), f"{ref_fp.as_posix()}"
        img = read_image(ref_fp, dtype='uint8')
        agg_img = np.zeros((len(folders_to_agg), img.shape[0], img.shape[1]), dtype='uint8')
        agg_img[0, :, :] = img
        
        for i, f in enumerate(folders_to_agg[1:]):
            fp = f / tile_name
            assert fp.exists(), f"{fp.as_posix()}"
            img = read_image(fp, dtype='uint8')
            agg_img[i + 1, :, :] = img
           
        agg_img = np.sum(agg_img, axis=0)
        agg_img = (agg_img >= (len(folders_to_agg) - 1)).astype('uint8')
    
        with rio.open(ref_fp, 'r') as src:
            profile = src.profile
            profile.update(dtype=rio.uint8, count=1)            
            filepath = (output_path / tile_name).as_posix()
            with rio.open(filepath, 'w', **profile) as dst:
                dst.write(agg_img.astype(rio.uint8), 1)


def run(input_paths):

    master_folders = sorted([p.name for p in input_paths[0].iterdir() if p.is_dir()])
    
    list_folders_to_agg = [[input_paths[0] / f, ] for f in master_folders]
    # assert that other have same folders
    for input_path in input_paths[1:]:
        folders = sorted([p.name for p in input_path.iterdir() if p.is_dir()])
        assert folders == master_folders, "{} vs {}".format(master_folders, folders)
        for j, f in enumerate(folders):
            list_folders_to_agg[j].append(input_path / f)

    output_path = mlflow.get_artifact_uri()
    output_path = Path(output_path)

    for folders_to_agg in tqdm.tqdm(list_folders_to_agg):        
        out_fp = output_path / folders_to_agg[0].name
        aggregate_and_save(folders_to_agg, out_fp)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Create ensembled predictions")
    parser.add_argument("input_paths", type=str, help="List of input paths to predictions as str object with ; as delimiter")

    args = parser.parse_args()
    input_paths = args.input_paths
    input_paths = input_paths.split(";")
    assert len(input_paths) > 0, "{}".format(len(input_paths))
    input_paths = [Path(p) for p in input_paths]
    for i in input_paths:
        assert i.exists(), f"Not found {i.as_posix()}"

    run(input_paths)
