import numpy as np
from osgeo import gdal
import os


def extract_by_shp(in_shp_path, inputpath, outputpath):
    input_raster = gdal.Open(inputpath)
    result = gdal.Warp(
        outputpath,
        input_raster,
        format='GTiff',
        cutlineDSName=in_shp_path,
        cropToCutline=True,
        dstNodata=np.nan
    )
    del result


def extract_basin():
    for basin in range(168):
        for year in range(22):
            for month in range(12):
                in_shp_path = r"data\site_info\area\area_weight"+'\\'+ str(basin+1) + ".shp"
                inputpath = r"results\global\SAI\tiff" +'\\'+ str(2000+year) + str(month+1).zfill(2) + ".tif"
                outputpath = r"results\basin\SAI" +'\\' + str(2000+year) + '\\' + str(month+1).zfill(2) + '\\' + str(basin+1) + ".tif"
                extract_by_shp(in_shp_path, inputpath, outputpath)
        print(""+str(basin+1)+"basin finish!")

extract_basin()