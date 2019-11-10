import argparse

from pathlib import Path

import tqdm

from org_tools.rasterize import rasterize_shp


def run(input_path, output_path):

    for shp_fp in tqdm.tqdm(list(input_path.rglob("*.shp"))):
        shp_parent = shp_fp.parent.stem
        output = output_path / shp_parent
        output.mkdir(parents=True, exist_ok=True)
        for ref_fp in tqdm.tqdm(list(shp_fp.parent.rglob("*vh*.tif"))):
            name = ref_fp.stem + ".tif"
            output_fp = output / name
            if output_fp.exists():
                output_fp = output / (ref_fp.parent.name + "_" + name)
            rasterize_shp(ref_fp.as_posix(), shp_fp.as_posix(), output_fp.as_posix())


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Generate mask from shape file with reference image")
    parser.add_argument("input_path", type=str, help="Input path to search for shape files")
    parser.add_argument("output_path", type=str, help="Output path")

    args = parser.parse_args()
    input_path = Path(args.input_path)
    assert input_path.exists(), f"Not found {input_path}"

    output_path = Path(args.output_path)
    assert not output_path.exists(), "Output should be a new folder"
    output_path.mkdir(parents=True)

    run(input_path, output_path)
