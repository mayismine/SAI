from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn import metrics
import os

rainday = str(7)
path = r'experiment\exp1\maxmin'
test_path = r'experiment\exp1\test_rich'
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
X = np.array(dataset[['WS', 'Ta', 'RH', 'Press', 'SW_IN','LAI', 'SIF', 'Month','acc_rain'+rainday,'DEM','GRA','CLAY','SAND','SILT','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','SM']])
Y = np.array(dataset[['ET']])
X_test = np.array(testdata[['WS', 'Ta', 'RH', 'Press', 'SW_IN','LAI', 'SIF', 'Month','acc_rain'+rainday,'DEM','GRA','CLAY','SAND','SILT','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','SM']])
Y_test = np.array(testdata[['ET']])
regr = RandomForestRegressor(n_estimators=tree_num,random_state=100)

regr.fit(X,Y.ravel())
y_pred = regr.predict(X_test)
MSE = metrics.mean_squared_error(Y_test,y_pred)
MAE = metrics.mean_absolute_error(Y_test,y_pred)
RMSE = np.sqrt(MSE)
R2 = metrics.r2_score(Y_test,y_pred)
NSE = metrics.r2_score(y_pred,Y_test)
print(RMSE,R2)
calculate_NSE(RMSE,Y_test,y_pred)
print('==================')


out = r'results\exp1\RF\test_rich'
for file in os.listdir(test_path):
    filepath = os.path.join(test_path, file)
    file_data = pd.read_csv(filepath)
    ET = np.array(file_data[['WS', 'Ta', 'RH', 'Press', 'SW_IN','LAI', 'SIF', 'Month','acc_rain'+rainday,'DEM','GRA','CLAY','SAND','SILT','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','SM']])
    ET_pred = regr.predict(ET)
    writedata = pd.DataFrame(ET_pred)
    writer = pd.ExcelWriter((out + '/' + file).replace('.csv', '.xlsx'))
    writedata.to_excel(writer, 'sheet_1', float_format='%.5f')
    writer.save()


