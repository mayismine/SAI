import torch
import numpy as np
import pandas as pd
from code.SAI import MSMT_LE
import os

def data_norm(data):
    maxlist = []
    minlist = []
    for i in range(data.shape[1]):
        input_data = data[:,i:i+1]
        max = np.max(input_data)
        min = np.min(input_data)
        maxlist.append(max)
        minlist.append(min)
        input_data = (input_data-min)/(max-min)
        data[:,i:i+1] = input_data
    return maxlist,minlist,data

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

data = get_data(r'experiment\exp3\maxmin')
datatrainx = np.array(data[["acc_rain7","WS","Ta", "Press", "RH", "SW_IN",'SM','LAI','SIF',"Month",'DEM','GRA','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT']])
maxx,minx,datatrainx = data_norm(datatrainx)
datatrainy = np.array(data[["ET",'H']])
maxy,miny,datatrainy = data_norm(datatrainy)


in_features = 9
sparameters = 7
num_heads = 3
hidden_features= 48
parameter_features = 20
gate_num = 6
control_num = 16
gate_True = True
attention_layers = 10
linear_drop = 0.1

model = MSMT_LE(in_features,hidden_features,attention_layers,num_heads,gate_num,linear_drop,control_num,gate_True)

model_dict = model.state_dict()
model_path = r"logs\Exp3_SAI.pth"
pretrained_dict = torch.load(model_path)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.eval()

root = r'experiment\exp3\test'
out = r'results\exp3\SAI'
for file in os.listdir(root):
    filepath = os.path.join(root, file)
    file_data = pd.read_csv(filepath)
    print(file)
    datatestx = np.array(file_data[["acc_rain7","WS","Ta", "Press", "RH", "SW_IN",'SM','LAI','SIF',"Month",'DEM','GRA','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT']]).astype(float)
    datatesty = np.array(file_data[["ET"]]).astype(float)
    for i in range(datatestx.shape[1]):
        input_data = datatestx[:, i:i + 1]
        input_data = (input_data - minx[i]) / (maxx[i] - minx[i])
        datatestx[:, i:i + 1] = input_data
    datapre = torch.FloatTensor(datatestx)
    results = []
    for i in range(datatestx.shape[0]):
        with torch.no_grad():
            # output = model(datapre[i:i + 1,:,:])
            output = model(datapre[i:i + 1, :])
            output = torch.squeeze(output, 0).numpy().tolist()
            results.append(output)
    results = np.array(results)
    print(results.shape)
    result = []
    for i in range(results.shape[1]):
        input_data_results = results[:, i:i + 1].reshape(-1)
        input_data_results = input_data_results * (maxy[i] - miny[i]) + miny[i]
        result.append(input_data_results)
    result = np.array(result)

    result = result.transpose()

    writedata = pd.DataFrame(result)
    writer = pd.ExcelWriter((out + '/' +file).replace('.csv','.xlsx'))
    writedata.to_excel(writer, 'sheet_1', float_format='%.5f')
    writer.save()

