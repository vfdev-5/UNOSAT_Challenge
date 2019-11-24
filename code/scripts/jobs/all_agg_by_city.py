# Script to aggregate predictions by city


import argparse

from pathlib import Path

import numpy as np

import tqdm

import rasterio as rio

from dataflow.io_utils import read_image


unique_cities_map = {
    "38RNV": "38RNV_Samawah",
    "38SLD": "38SLD_Tikrit",
    "38SMB": "38SMB_Bagdad",
    "38SME": "38SME_Kirkouk",
}


def aggregate_and_save(folders_to_agg, filepath):
    path = folders_to_agg[0]
    ref_image_fp = path / "merged.tif"
    img = read_image(ref_image_fp, dtype='uint8')

    agg_img = np.zeros((len(folders_to_agg), img.shape[0], img.shape[1]), dtype='uint8')
    agg_img[0, :, :] = img
    for i, path in enumerate(folders_to_agg[1:]):
        mask_fp = path / "merged.tif"
        img = read_image(mask_fp, dtype='uint8')
        agg_img[i + 1, :, :] = img
    
    agg_img = np.sum(agg_img, axis=0)
    agg_img = (agg_img >= (len(folders_to_agg) - 1)).astype('uint8')

    with rio.open(ref_image_fp, 'r') as src:
        profile = src.profile
        profile.update(dtype=rio.uint8, count=1)

        with rio.open(filepath, 'w', **profile) as dst:
            dst.write(agg_img.astype(rio.uint8), 1)


def run(input_path):

    folders = [p for p in input_path.iterdir()]
    for c in tqdm.tqdm(unique_cities_map):        
        folders_to_agg = [f for f in folders if c in f.as_posix()]
        out_fp = input_path / (unique_cities_map[c] + ".tif")
        aggregate_and_save(folders_to_agg, out_fp)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Generate tiles stats")
    parser.add_argument("input_path", type=str, help="Input path to predictions")

    args = parser.parse_args()
    input_path = Path(args.input_path)
    assert input_path.exists(), f"Not found {input_path}"

    run(input_path)
