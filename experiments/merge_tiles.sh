#!/bin/bash
set -e

echo "- Merge tiles predictions into a single mask image -"

input_path=$1

echo ""
echo "Input path: $input_path"
echo ""

echo ""
echo "Merge:"
echo ""

cd $input_path && for i in `ls .`; do gdal_merge.py -o $i/merged.tif `find $i -name tile*.tif`; done

echo ""
echo "Done"
echo ""

