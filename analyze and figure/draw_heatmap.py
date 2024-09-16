import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def R_2(X,Y):
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    sum5 = 0
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    for i in range(len(X)):
        sum1 = sum1 + (X[i]-X_mean)*(Y[i]-Y_mean)
        sum2 = sum2 + ((X[i]-X_mean)**2)
        sum3 = sum3 + ((Y[i] - Y_mean) ** 2)
        sum4 = sum4 + ((X[i]-Y[i])**2)
        sum5 = sum5 + ((X[i]-X_mean)**2)
    R2 = (sum1/((sum2 ** 0.5)*(sum3 ** 0.5))) ** 2
    return R2

def RMSE(X,Y):
    return np.sqrt(np.sum(np.power(X - Y,2)) / len(X))

def heatmap_basin(basin,num):
    RF_path = r'results\basin\common\RF_common.xlsx'  #_common
    Attention_path = r'results\basin\common\Attention_common.xlsx'
    MTMS_path = r'results\basin\common\MTMS_common.xlsx'
    SAI_path = r'results\basin\common\SAI_common.xlsx'
    FLUXCOM_path = r'results\basin\common\Fluxcom_common.xlsx'
    GLEAM_path = r'results\basin\common\GLEAM_common.xlsx'

    datas = []
    Attention_data = np.array(pd.read_excel(Attention_path).loc[basin,'200205':'201612']).flatten()  #'200205':'201612'
    MTMS_data = np.array(pd.read_excel(MTMS_path).loc[basin,'200205':'201612']).flatten()
    RF_data = np.array(pd.read_excel(RF_path).loc[basin,'200205':'201612']).flatten()
    SAI_data = np.array(pd.read_excel(SAI_path).loc[basin,'200205':'201612']).flatten()
    GLEAM_data = np.array(pd.read_excel(GLEAM_path).loc[basin,'200205':'201612']).flatten()
    FLUXCOM_data = np.array(pd.read_excel(FLUXCOM_path).loc[basin,'200205':'201612']).flatten()


    datas.append(RF_data)
    datas.append(Attention_data)
    datas.append(MTMS_data)
    datas.append(SAI_data)
    datas.append(GLEAM_data)
    datas.append(FLUXCOM_data)

    heat_maps = np.full((num,num),0,dtype=float)

    for i in range(num):
        for j in range(num):
            X = datas[i]
            Y = datas[j]
            R2 = R_2(X,Y)
            Rmse = RMSE(X,Y)
            heat_maps[i,j] = Rmse

    return heat_maps

plt.figure(figsize=(8, 6))
columns = ['RF','SA','MTMS','SAI','GLEAM','FLUXCOM']
# basin = [4,20,21,30,35]
basin = [0]
heat_maps = heatmap_basin(basin,6)
ax = sns.heatmap(heat_maps, annot=True, fmt='.2f',xticklabels=columns,yticklabels=columns,cmap='Reds')
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.show()