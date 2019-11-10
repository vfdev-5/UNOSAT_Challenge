from osgeo import gdal
from osgeo import osr, ogr, gdalconst


def array2raster(newRasterfn, dataset, array, dtype):
    """
    save GTiff file from numpy.array
    input:
        newRasterfn: save file name
        dataset : original tif file
        array : numpy.array
        dtype: Byte or Float32.
    """
    cols = array.shape[1]
    rows = array.shape[0]
    originX, pixelWidth, b, originY, d, pixelHeight = dataset.GetGeoTransform()

    driver = gdal.GetDriverByName('GTiff')

    # set data type to save.
    GDT_dtype = gdal.GDT_Unknown
    if dtype == "Byte":
        GDT_dtype = gdal.GDT_Byte
    elif dtype == "Float32":
        GDT_dtype = gdal.GDT_Float32
    else:
        print("Not supported data type.")

    # set number of band.
    if array.ndim == 2:
        band_num = 1
    else:
        band_num = array.shape[2]

    outRaster = driver.Create(newRasterfn, cols, rows, band_num, GDT_dtype)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    # Loop over all bands.
    for b in range(band_num):
        outband = outRaster.GetRasterBand(b + 1)
        # Read in the band's data into the third dimension of our array
        if band_num == 1:
            outband.WriteArray(array)
        else:
            outband.WriteArray(array[:,:,b])

    # setteing srs from input tif file.
    prj = dataset.GetProjection()
    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


def rasterize_shp(template_raster, shp, output,
                  dtype=gdal.GDT_Byte,
                  options=("ATTRIBUTE=DN", "ALL_TOUCHED=TRUE"),
                  nodata_val=0):

    model_dataset = gdal.Open(template_raster)
    shape_dataset = ogr.Open(shp)
    shape_layer = shape_dataset.GetLayer()
    mem_drv = gdal.GetDriverByName('MEM')
    mem_raster = mem_drv.Create(
        '',
        model_dataset.RasterXSize,
        model_dataset.RasterYSize,
        1,
        dtype
    )
    mem_raster.SetProjection(model_dataset.GetProjection())
    mem_raster.SetGeoTransform(model_dataset.GetGeoTransform())
    mem_band = mem_raster.GetRasterBand(1)
    mem_band.Fill(nodata_val)
    mem_band.SetNoDataValue(nodata_val)

    err = gdal.RasterizeLayer(
        mem_raster,
        [1],
        shape_layer,
        None,
        None,
        [1],
        options
    )
    assert err == gdal.CE_None
    label_array = mem_raster.ReadAsArray()
    label_array[label_array == 255] = 1
    array2raster(output, model_dataset, label_array, 'Byte')


if __name__ == "__main__":

    img_raster = '/data_dir/data_dir/s1tiling_S1A_IW_GRDH_1SDV_20150102T145957_20150102T150022_003994_004CF1_CDEF/38SNE/s1a_38SNE_vh_ASC_20150102t145957.tif'
    shape = '/data_dir/data_dir/38SNE_Souleimaniye.shp'
    output = '/data_dir/data_dir/shit.tif'
    rasterize_shp(img_raster, shape, output, gdal.GDT_Byte, ["ATTRIBUTE=DN", "ALL_TOUCHED=TRUE"])
