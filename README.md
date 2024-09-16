# Physics-informed deep-learning model for mitigating spatiotemporal imbalances in FLUXNET2015 global evapotranspiration data

These are code and data implementations.

### Prerequisites

Make sure the version of python is not too low (python 3.6 recommended) and some necessary dependencies::
```
model : torch,sklearn
graph : matplotlib,seaborn,pandas,numpy
analyze : shapely,pandas,numpy,arcpy,gdal
```

### 1. Data Preparation
Download data from :
- site data : [FLUXNET2015](https://fluxnet.org/) , [other site](https://data.tpdc.ac.cn/home);
- remete sensing data : [leaf area index(LAI)](https://doi.org/10.1175/BAMS-D-18-0341.1) , [solar-induced fluorescence(SIF)](https://doi.org/10.3390/rs11050517) , [digital elevation model(DEM)](https://doi.org/10.1029/2002GL016643) , [soil moisture(SM)](https://doi.org/10.1038/s41597-023-01991-w) , [soil texture](https://doi.org/10.1002/2013MS000293);
- global data : [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means?tab=overview) , [GLDAS](https://ldas.gsfc.nasa.gov/gldas).

After data cleaning, interpolation, extraction and other operations, put all data into a seperate folder `./data` 

For each experiment, we have placed relevant training, validation, testing, and other data:
```
experiment
├── exp1
│   └── maxmin
│   └── test_lack
│   └── test_rich
│   └── train
│   └── validation
├── exp2
│   └── maxmin
│   └── test
│   └── train
│   └── validation
├── exp3
│   └── maxmin
│   └── test
│   └── train
│   └── validation
└── global
    └── maxmin
    └── train
    └── val
```
Notably : 
- the data above is the data used in the paper, and the data set can be customized.
- For the data of global and basin part, please download them yourself due to the large amount of reanalysis data

### 2. Training

The training scripts are located in the `./code` directory. 

**To train a model using the experimental dataset:**
1. Modify the `path` in the file at the end of `_train.py`.
2. training and test codes of RF model are in the same file.

**Upon completion of the training:**
- The trained model will be saved in the `./logs` directory.
- make predictions in the file at the end of `_pre.py`.
- ET results for four models are located in `./results`.

**Notably**

There is uncertainty in the training process of the model, so the training will produce inconsistencies with the results of the paper, but the error is small and does not affect all the conclusions.

Some training weights are saved into `. /logs`, which can be used to reproduce exactly the same results for most results of this paper.

### Analyzing
All of the analysis graphs and code presented in this paper can be found in the `./analyze and figure` directory. 

**SHAP**

A detailed implementation of the SHAP explainability method can be found at this [website](https://shap.readthedocs.io/en/latest/index.html).

After training the model, the results of SHAP value analysis are displayed in the `./analyze and figure/SHAP` directory in `.xlsx` format , such as:
```
SHAP
└── MTMS
    └── PDP.xlsx
    └── PDP_LAI_SM.xlsx
    └── PDP_LAI_SM_ori_data.xlsx
    └── PDP_ori_data.xlsx
    └── SHAP.xlsx
    └── SHAP_ori_data.xlsx
    └── SHAP_row.xlsx

```
Notably:
Because there is no operation to export SHAP values in SHAP, modify its source code to export is needed. For this paper, the results are all in the `./analyze and figure/SHAP` directory.

**basin**
- The basin results are based on common pixel (the missing value is different for each product) and area-weighted (each pixel represents a different area) processing.
- The common pixels for all products (models) in this paper can be found in the `./analyze and figure/common_pixel_global` directory.
- The TCH method (the results are in the `./analyze and figure/TCH` directory) is affected by the number of products, but using only SAI, GLEAM and FLUXCOM, the uncertainty of SAI proposed by us is still the lowest in the data-poor basin.
- The results of the MK test are in the `./analyze and figure/MK` directory.
### Citation
If you find this repo useful, please cite our paper as follows:

```
@article{,
  title=,
  author=,
  journal=,
  year=
}
```

### Contact
If you have any questions, please contact xutr@bnu.edu.cn or submit an issue.

### Acknowledgement
We appreciate the following repo for their code:
- https://github.com/Kyubyong/transformer
- https://github.com/xutr-bnu/TCH_method
