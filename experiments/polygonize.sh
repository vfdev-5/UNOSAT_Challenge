#!/bin/bash
set -e

echo "- Polygonize predictions into shapefiles -"

input_path=$1

echo ""
echo "Input path: $input_path"
echo ""

echo ""
echo "Polygonize:"
echo ""

cd $input_path && for i in "38RNV_Samawah" "38SLD_Tikrit" "38SMB_Bagdad" "38SME_Kirkouk"; do gdal_polygonize.py "$i.tif" -f "ESRI Shapefile" "$i.shp" $i id; done

echo ""
echo "Clean temporary .tif :"
echo ""

cd $input_path && for i in "38RNV_Samawah" "38SLD_Tikrit" "38SMB_Bagdad" "38SME_Kirkouk"; do rm "$i.tif"; done

echo ""
echo "Done"
echo ""

