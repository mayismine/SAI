import os

from osgeo import gdal
import pandas as pd
import numpy as np

datas = np.full((50,204),np.nan)
index = []
for basin in range(50):
    columns = []
    area_weight_path = r'data\site_info\area\area_weight' + '\\' + str(basin+1) + '.tif'
    area_datas = gdal.Open(area_weight_path)
    area_cols = area_datas.RasterXSize
    area_rows = area_datas.RasterYSize
    area_band = area_datas.GetRasterBand(1)
    area_data = area_band.ReadAsArray(0, 0, area_cols, area_rows)
    for year in range(17):
        for month in range(12):
            path = r'results\basin\SAI' + '\\' + str(2000+year) + '\\' + str(month+1).zfill(2) + '\\' + str(basin+1) + '.tif'
            # t2m = r'F:\postgraduate\graduation_project\results\Fluxcom\t2m_0.25'+ '\\' + str(2000+year) + '\\' + str(month+1).zfill(2) + '\\' + str(basin+1) + '.tif'
            area_common = r'results\basin\basin_common' + '\\'  + str(2000+year) + '\\' + str(month+1).zfill(2) + '\\' + str(basin+1) + '.tif'
            ds = gdal.Open(path)
            cols = ds.RasterXSize
            rows = ds.RasterYSize
            band = ds.GetRasterBand(1)
            data = band.ReadAsArray(0 ,0 ,cols ,rows)

            # t2m_ds = gdal.Open(t2m)
            # t2m_cols = t2m_ds.RasterXSize
            # t2m_rows = t2m_ds.RasterYSize
            # t2m_data = t2m_ds.GetRasterBand(1).ReadAsArray(0 ,0 ,t2m_cols ,t2m_rows)

            common_ds = gdal.Open(area_common)
            common_cols = common_ds.RasterXSize
            common_rows = common_ds.RasterYSize
            common_data = common_ds.GetRasterBand(1).ReadAsArray(0 ,0 ,common_cols ,common_rows)

            total = 0
            num = 0
            weights = 0
            # if data.shape[0]>area_data.shape[0] or data.shape[1]>area_data.shape[1]:
            #     data = data[0:area_data.shape[0],0:area_data.shape[1]]
            for row in range(data.shape[0]):
                for col in range(data.shape[1]):
                    value = data[row][col]
                    weight = area_data[row][col]
                    common = common_data[row][col]
                    # if row >= t2m_rows or col >= t2m_cols:
                    #     t2m = 25
                    # else:
                    #     t2m = t2m_data[row][col]
                    if value > 0:# and t2m>-1000 and weight>-10 and common>-1:
                        total = total + value * weight * common #/ (2.5 - 0.00237 * t2m)
                        num = num + 1
                        weights = weights + weight
            if num != 0:
                mean = total / weights
            if num == 0:
                mean = 0
            datas[basin,year*12+month] = mean
            columns.append(str(2000+year)+str(month+1).zfill(2))
    index.append(str(basin+1))

frame = pd.DataFrame(datas,index=index,columns=columns)
frame.to_excel(r'results\basin\common\SAI_common.xlsx')


