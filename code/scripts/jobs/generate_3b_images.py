import argparse

from pathlib import Path
from joblib import Parallel, delayed

import tqdm

import gdal


def write_3b_image(p1, p2, p_out):
    src1 = gdal.Open(p1, gdal.GA_ReadOnly)
    assert src1 is not None, "Failed to read the file {}".format(p1)
    src2 = gdal.Open(p2, gdal.GA_ReadOnly)
    assert src2 is not None, "Failed to read the file {}".format(p2)

    assert src1.RasterXSize == src2.RasterXSize
    assert src1.RasterYSize == src2.RasterYSize
    assert src1.RasterCount == src2.RasterCount == 1
    assert src1.GetGeoTransform() == src2.GetGeoTransform()
    assert src1.GetProjection() == src2.GetProjection()

    h = src1.RasterYSize
    w = src1.RasterXSize
    c = 3

    dst = gdal.GetDriverByName('GTiff').Create(p_out, w, h, c, gdal.GDT_Float32)

    dst.SetGeoTransform(src1.GetGeoTransform())
    dst.SetMetadata(src1.GetMetadata())
    dst.SetProjection(src1.GetProjection())

    d1 = src1.ReadAsArray()
    d2 = src2.ReadAsArray()

    dst.GetRasterBand(1).WriteArray(d1)
    dst.GetRasterBand(2).WriteArray(d2)
    dst.GetRasterBand(3).WriteArray((d1 + d2) * 0.5)

    dst = None


def worker_task(p1, p2, output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    p_out = output_path / p1.name.replace("_vh_", "_3b_")

    if p_out.exists():
        p_out = output_path / (p1.parent.name + "_" + p1.name.replace("_vh_", "_3b_"))

    write_3b_image(p1.as_posix(), p2.as_posix(), p_out.as_posix())


def run(input_path, output_path):
    from pathlib import Path

    vh_images = list(input_path.rglob("*vh*.tif"))
    vv_images = [Path(p.as_posix().replace("_vh_", "_vv_")) for p in vh_images]

    for p1, p2 in zip(vh_images, vv_images):
        assert p1.exists()
        assert p2.exists()

    i = len(input_path.parts)
    with Parallel(n_jobs=8) as parallel:
        parallel(delayed(worker_task)(p1, p2, output_path / p1.parts[i])
                 for p1, p2 in tqdm.tqdm(zip(vh_images, vv_images), total=len(vh_images)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Generate 3-bands (vh, vv, (vh + vv)/2) images from vh/vv original images")
    parser.add_argument("input_path", type=str, help="Input path to search for shape files")
    parser.add_argument("output_path", type=str, help="Output path")

    args = parser.parse_args()
    input_path = Path(args.input_path)
    assert input_path.exists(), f"Not found {input_path}"

    output_path = Path(args.output_path)
    assert not output_path.exists(), "Output should be a new folder"
    output_path.mkdir(parents=True)

    run(input_path, output_path)
