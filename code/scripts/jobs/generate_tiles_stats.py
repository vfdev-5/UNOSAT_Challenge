# Script to generate stats csv from tiles
# Output CSV will contain the following fields
# index, img_min, img_max, img_mean, img_std, target_ratio, fold_index, skip, image_path

import argparse

from pathlib import Path
import pandas as pd

import tqdm

from dataflow.datasets import UnoSatTiles
from dataflow.io_utils import read_image

folds = ["38SLF", "38RPV", "38SMA", "38SNE"]


def run(input_path, output_path):

    train_dataset = UnoSatTiles(input_path)

    data = [None] * len(train_dataset)

    for i, dp in tqdm.tqdm(enumerate(train_dataset), total=len(train_dataset)):
        img = read_image(dp['image'], dtype='float32')
        mask = read_image(dp['mask'], dtype='uint8')

        img_ravel = img.reshape((-1, img.shape[-1]))
        img_min = tuple(img_ravel.min(axis=0))
        img_max = tuple(img_ravel.max(axis=0))
        img_mean = tuple(img_ravel.mean(axis=0))
        img_std = tuple(img_ravel.std(axis=0))

        target_ratio = (mask > 0).sum() / (mask.shape[0] * mask.shape[1])

        fold_index = [f in dp['image'].as_posix() for f in folds].index(True)
        skip = sum(img_min) == sum(img_max)

        if img_min[0] < -10000:
            skip = True

        data[i] = [i, img_min, img_max, img_mean, img_std, target_ratio, fold_index, skip, dp['image']]

    df = pd.DataFrame(data, columns=["index", "img_min", "img_max", "img_mean", "img_std",
                                     "target_ratio", "fold_index", "skip", "img_path"])
    output_fp = output_path / "tile_stats.csv"
    df.to_csv(output_fp, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Generate tiles stats")
    parser.add_argument("input_path", type=str, help="Input path to tiles : images and masks folders")
    parser.add_argument("output_path", type=str, help="Output path")

    args = parser.parse_args()
    input_path = Path(args.input_path)
    assert input_path.exists(), f"Not found {input_path}"
    output_path = Path(args.output_path)
    assert output_path.exists(), f"Not found {output_path}"

    run(input_path, output_path)
