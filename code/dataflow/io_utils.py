from typing import Union, Optional, Sequence

from pathlib import Path
import numpy as np

import gdal

import rasterio as rio
from rasterio.windows import Window, intersection
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning

import warnings
warnings.simplefilter(action='ignore', category=NotGeoreferencedWarning)


def read_image(path: Union[str, Path],
               band_indices: Optional[Union[Sequence[int]]] = None,
               dtype: Optional[str] = None):
    """Read image tif file and apply the usual transformation

    Args:
        path: path of the tif file
        band_indices (list of integers, optional): indices of the band with an offset of 1
        dtype (str): output data type

    Return:
        A numpy array representation of the tif file
    """
    if isinstance(path, Path):
        path = path.as_posix()

    f = gdal.Open(path, gdal.GA_ReadOnly)
    assert f is not None, "Failed to read the file {}".format(path)

    if band_indices is None:
        band_indices = [i + 1 for i in range(f.RasterCount)]

    assert isinstance(band_indices, (list, tuple))

    if dtype is None:
        dtype = 'float32'

    output = np.zeros((f.RasterYSize, f.RasterXSize, len(band_indices)), dtype=dtype)
    for i, index in enumerate(band_indices):
        band = f.GetRasterBand(index)
        output[:, :, i] = band.ReadAsArray()

    if output.shape[-1] == 1:
        output = output.squeeze(-1)
    return output


def imwrite_rasterio(filepath, data, **kwargs):
    """
    Method to write image to file using Rasterio

    Args:
      filepath: output filepath
      data: ndarray with data of shape (h, w, c)
      kwargs: additional kwargs inserted to `rasterio.open`.

    Returns:

    """
    assert isinstance(data, np.ndarray) and data.ndim == 3, "Data should be a ndarray of shape (h, w, c)"
    h, w, c = data.shape
    if 'driver' not in kwargs:
        kwargs['driver'] = get_driver_from_extension(Path(filepath).suffix)
    with rio.open(filepath, 'w', height=h, width=w, count=c,
                  dtype=data.dtype, **kwargs) as dst:
        dst.write(data.transpose([2, 0, 1]))


def read_data_rasterio(rio_dataset, src_rect=None, dst_width=None, dst_height=None,
                       nodata_value=np.uint8(0), dtype=None, select_bands=None,
                       resampling=Resampling.nearest):
    """
    Method to read data from rasterio dataset

    Args:
      rio_dataset: instance of rasterio.io.DatasetReader
      src_rect: is source extent in pixels : [x,y,w,h] where (x,y) is top-left corner. Can be None and whole image
        extent is used. (Default value = None)
      dst_width: is the output array width. Can be None and src_rect[2] (width) is used. (Default value = None)
      dst_height: is the output array heigth. Can be None and src_rect[3] (height) is used. (Default value = None)
      nodata_value: value to fill out of bounds pixels with. (Default value = np.uint8(0)
      dtype: force type of returned numpy array
      select_bands: tuple of band indices (zero-based) to select from dataset, e.g. [0, 3, 4].
      resampling: rasterio resampling method (see rasterio.enums.Resampling)

    Returns:
      numpy.ndarray
    """
    assert isinstance(rio_dataset, rio.io.DatasetReader), \
        "Argument rio_dataset should be instance of rasterio.io.DatasetReader"
    assert rio_dataset.count > 0, "Dataset has no bands"

    if select_bands is not None:
        assert isinstance(select_bands, list) or isinstance(select_bands, tuple), \
            "Argument select_bands should be a tuple or list"
        available_bands = list(range(rio_dataset.count))
        for index in select_bands:
            assert index in available_bands, \
                "Index {} from select_bands is outside of available bands: {}".format(index, available_bands)

    if src_rect is None:
        src_req_extent = Window(0, 0, rio_dataset.width, rio_dataset.height)
        src_extent = [src_req_extent.col_off, src_req_extent.row_off, src_req_extent.width, src_req_extent.height]
    else:
        src_req_extent = intersection(Window(*src_rect), Window(0, 0, rio_dataset.width, rio_dataset.height))
        src_extent = src_rect

    if src_req_extent is None:
        print('Source request extent is None')
        return None

    if dst_width is None and dst_height is None:
        dst_extent = [src_extent[2], src_extent[3]]
    elif dst_height is None:
        h = int(dst_width * src_extent[3] * 1.0 / src_extent[2])
        dst_extent = [dst_width, h]
    elif dst_width is None:
        w = int(dst_height * src_extent[2] * 1.0 / src_extent[3])
        dst_extent = [w, dst_height]
    else:
        dst_extent = [dst_width, dst_height]

    scale_x = dst_extent[0] * 1.0 / src_extent[2]
    scale_y = dst_extent[1] * 1.0 / src_extent[3]
    req_scaled_w = int(min(np.ceil(scale_x * src_req_extent.width), dst_extent[0]))
    req_scaled_h = int(min(np.ceil(scale_y * src_req_extent.height), dst_extent[1]))

    r = [int(np.floor(scale_x * (src_req_extent.col_off - src_extent[0]))),
         int(np.floor(scale_y * (src_req_extent.row_off - src_extent[1]))),
         req_scaled_w,
         req_scaled_h]

    band_indices = range(rio_dataset.count) if select_bands is None else select_bands
    nb_bands = len(band_indices)

    if dtype is None:
        dtypes = rio_dataset.dtypes + (type(nodata_value),)
        dtype_sizes = [np.dtype(d).itemsize for d in dtypes]
        datatype = np.dtype(dtypes[np.argmax(dtype_sizes)])
    else:
        datatype = dtype

    out = np.empty((dst_extent[1], dst_extent[0], nb_bands), dtype=datatype)
    out.fill(nodata_value)

    band_indices = [b + 1 for b in band_indices]
    data = rio_dataset.read(indexes=band_indices,
                            out_shape=(len(band_indices), r[3], r[2]),
                            window=src_req_extent,
                            resampling=resampling)
    out[r[1]:r[1] + r[3], r[0]:r[0] + r[2], :] = np.transpose(data, [1, 2, 0])
    return out


# Until the issue is not solved : https://github.com/mapbox/rasterio/issues/265
# Mapping between extension and GDAL driver
EXTENSIONS_GDAL_DRIVER_CODE_MAP = {
    'asc': 'AAIGrid',
    'blx': 'BLX',
    'bmp': 'BMP',
    'bt': 'BT',
    'dat': 'ZMap',
    'dem': 'USGSDEM',
    'gen': 'ADRG',
    'gif': 'GIF',
    'gpkg': 'GPKG',
    'grd': 'NWT_GRD',
    'gsb': 'NTv2',
    'gtx': 'GTX',
    'hdr': 'MFF',
    'hf2': 'HF2',
    'hgt': 'SRTMHGT',
    'img': 'HFA',
    'jpg': 'JPEG',
    'kro': 'KRO',
    'lcp': 'LCP',
    'map': 'PCRaster',
    'mbtiles': 'MBTiles',
    'mpr/mpl': 'ILWIS',
    'ntf': 'NITF',
    'pix': 'PCIDSK',
    'png': 'PNG',
    'pnm': 'PNM',
    'rda': 'R',
    'rgb': 'SGI',
    'rst': 'RST',
    'rsw': 'RMF',
    'sdat': 'SAGA',
    'sqlite': 'Rasterlite',
    'ter': 'Terragen',
    'tif': 'GTiff',
    'vrt': 'VRT',
    'xpm': 'XPM',
    'xyz': 'XYZ'
}


def get_driver_from_extension(ext):
    ext = ext.replace(".", "")
    if ext in EXTENSIONS_GDAL_DRIVER_CODE_MAP:
        return EXTENSIONS_GDAL_DRIVER_CODE_MAP[ext]
    return None
