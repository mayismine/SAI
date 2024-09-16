from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn import metrics
import os


path = r'experiment\global\train'
test_path = r'experiment\global\val'
tree_num = 100

def get_data(path):
    list = []
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        filedata = pd.read_csv(filepath)
        list.append(filedata)
    data = list[0]
    for i in range(len(list)):
        if i == 0:
            continue
        data = pd.concat([data, list[i]])
    return data

def calculate_NSE(RMSE,X,Y):
    NSE1 = 1 - (RMSE ** 2 / np.var(X))
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
    NSE2 = 1 - (sum4 / sum5)
    print(NSE1,NSE2,R2)

dataset = get_data(path)
testdata = get_data(test_path)
# nodaratest = get_data(nodata_test_path)
# extremetest = get_data(extreme_test_path)
X = np.array(dataset[['WS', 'Ta', 'RH', 'P', 'SW_IN','LAI', 'SIF', 'Month','Rain','DEM','GRA','CLAY','SAND','SILT','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','SM']])
# X = np.array(dataset[['WS', 'Ta', 'RH', 'Press', 'SW_IN','LAI', 'SIF', 'Month','acc_rain'+rainday,'DEM','GRA','CLAY','SAND','SILT','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','SM']])
# X = np.array(dataset[['WS','Ta','RH','Press','SW_IN','LAI','SIF','acc_rain7','SM','DEM','GRA','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT']])
Y = np.array(dataset[['ET']])
X_test = np.array(testdata[['WS', 'Ta', 'RH', 'P', 'SW_IN','LAI', 'SIF', 'Month','Rain','DEM','GRA','CLAY','SAND','SILT','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','SM']])
# X_test = np.array(testdata[['WS', 'Ta', 'RH', 'Press', 'SW_IN','LAI', 'SIF', 'Month','acc_rain'+rainday,'DEM','GRA','CLAY','SAND','SILT','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','SM']])
# X_test = np.array(testdata[['WS','Ta','RH','Press','SW_IN','LAI','SIF','acc_rain7','SM','DEM','GRA','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT']])
Y_test = np.array(testdata[['ET']])
# X_nodatatest = np.array(nodaratest[['WS', 'Ta', 'RH', 'Press', 'SW_IN','LAI', 'SIF', 'Month','acc_rain'+rainday,'DEM','GRA','CLAY','SAND','SILT','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','SM']])
# Y_nodatatest = np.array(nodaratest[['ET']])
# X_extremetest = np.array(extremetest[['WS', 'Ta', 'RH', 'Rain', 'Press', 'SW_IN', 'LAI', 'SIF', 'Month','DEM','GRA','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH']])
# Y_extremetest = np.array(extremetest[['ET']])
regr = RandomForestRegressor(n_estimators=tree_num,random_state=100)

regr.fit(X,Y.ravel())
# regr.fit(X_all,Y_all.ravel())
y_pred = regr.predict(X_test)
# y_nodata_pre = regr.predict(X_nodatatest)
# y_extreme_pre = regr.predict(X_extremetest)
MSE = metrics.mean_squared_error(Y_test,y_pred)
MAE = metrics.mean_absolute_error(Y_test,y_pred)
RMSE = np.sqrt(MSE)
R2 = metrics.r2_score(Y_test,y_pred)
NSE = metrics.r2_score(y_pred,Y_test)
print(RMSE,R2)
calculate_NSE(RMSE,Y_test,y_pred)
print('==================')

for i in range(22):
    start = 12 * i
    end = 12 * i + 12
    year = 2000 + i
    Rain = np.load(r'data\gloabl_forcing\tp.npy')[start:end,:,:].astype(np.float32)
    WS = np.load(r'data\gloabl_forcing\si10.npy')[start:end,:,:].astype(np.float32)
    Ta = np.load(r'data\gloabl_forcing\t2m.npy')[start:end,:,:].astype(np.float32)
    P = np.load(r'data\gloabl_forcing\sp.npy')[start:end,:,:].astype(np.float32)
    RH = np.load(r'data\gloabl_forcing\RH.npy')[start:end,:,:].astype(np.float32)
    SW_IN = np.load(r'data\gloabl_forcing\ssrd.npy')[start:end,:,:].astype(np.float32)
    SM = np.load(r'data\gloabl_forcing\swvl1.npy')[start:end,:,:].astype(np.float32)
    LAI = np.load(r'data\gloabl_forcing\LAI.npy')[start:end,:,:].astype(np.float32)
    SIF = np.load(r'data\gloabl_forcing\SIF.npy')[start:end,:,:].astype(np.float32)
    # Month = np.load(r'F:\postgraduate\graduation_project\tiff_data\Month_revise.npy')[start:end,:,:]
    DEM = np.load(r'data\gloabl_forcing\DEM.npy')[start:end,:,:].astype(np.float32)
    ENF = np.load(r'data\gloabl_forcing\ENF.npy')[start:end,:,:].astype(np.int32)
    EBF = np.load(r'data\gloabl_forcing\EBF.npy')[start:end,:,:].astype(np.int32)
    DBF = np.load(r'data\gloabl_forcing\DBF.npy')[start:end,:,:].astype(np.int32)
    MF = np.load(r'data\gloabl_forcing\MF.npy')[start:end,:,:].astype(np.int32)
    CSH = np.load(r'data\gloabl_forcing\CSH.npy')[start:end,:,:].astype(np.int32)
    OSH = np.load(r'data\gloabl_forcing\OSH.npy')[start:end,:,:].astype(np.int32)
    WSA = np.load(r'data\gloabl_forcing\WSA.npy')[start:end,:,:].astype(np.int32)
    SAV = np.load(r'data\gloabl_forcing\SAV.npy')[start:end,:,:].astype(np.int32)
    GRA = np.load(r'data\gloabl_forcing\GRA.npy')[start:end,:,:].astype(np.int32)
    WET = np.load(r'data\gloabl_forcing\WET.npy')[start:end,:,:].astype(np.int32)
    CRO = np.load(r'data\gloabl_forcing\CRO.npy')[start:end,:,:].astype(np.int32)
    SAND = np.load(r'data\gloabl_forcing\SAND.npy')[start:end,:,:].astype(np.float32)
    CLAY = np.load(r'data\gloabl_forcing\CLAY.npy')[start:end,:,:].astype(np.float32)
    SILT = np.load(r'data\gloabl_forcing\SILT.npy')[start:end,:,:].astype(np.float32)

    final_data = np.full((12,720,1440),np.nan)
    for month in np.arange(12):
        for col in np.arange(1440):
            input_data = np.full((720,24),np.nan)
            input_data[:,7] = Rain[month,:,col]
            input_data[:,0] = WS[month, :, col]
            input_data[:,1] = Ta[month, :, col]
            input_data[:,3] = P[month, :, col]
            input_data[:,2] = RH[month, :, col]
            input_data[:,4] = SW_IN[month, :, col]
            input_data[:,23] = SM[month, :, col]
            input_data[:,5] = LAI[month, :, col]
            input_data[:,6] = SIF[month, :, col]
            # input_data[:,9] = Month[month, :, col]
            input_data[:,8] = DEM[month, :, col]
            input_data[:,10] = SAND[month, :, col]
            input_data[:,9] = CLAY[month, :, col]
            input_data[:,11] = SILT[month, :, col]
            input_data[:,18] = ENF[month, :, col]
            input_data[:,15] = EBF[month, :, col]
            input_data[:,16] = DBF[month, :, col]
            input_data[:,17] = MF[month, :, col]
            input_data[:,22] = CSH[month, :, col]
            input_data[:,21] = OSH[month, :, col]
            input_data[:,13] = WSA[month, :, col]
            input_data[:,14] = SAV[month, :, col]
            input_data[:,12] = GRA[month, :, col]
            input_data[:,19] = WET[month, :, col]
            input_data[:,20] = CRO[month, :, col]
            input_nan = np.isnan(input_data).any(axis=1)
            input_data[np.where(np.isnan(input_data))] = -99
            output_data = regr.predict(input_data)
            output_data[input_nan] = np.nan
            final_data[month,:,col] = output_data
        print("第"+str(month+1)+"月已完成")
    np.save(r'global\RF\npy'+'\\'+str(year),final_data)

