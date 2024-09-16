import torch
import numpy as np
import pandas as pd
from task.ET.model.SA import Attention
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

index = ["acc_rain7","WS","Ta", "Press", "RH", "SW_IN","Month",'LAI','SIF','DEM','GRA','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT','SM']
# Notably: the position of the explanatory variables may bring some bias in the results, but it does not affect the final conclusion


max_min_data = get_data(r'experiment\exp1\maxmin')
datamaxmin = np.array(max_min_data[index]).astype(float) # rain
maxx,minx,_ = data_norm(datamaxmin)
datamaxminy = np.array(max_min_data[["ET"]]).astype(float)
maxy,miny,_ = data_norm(datamaxminy)

in_features = 25
sparameters = 7
num_heads = 3
depth = 1
hidden_features= 48
parameter_features = 20

attention_layers = 10
linear_drop = 0.1

model = Attention(in_features,hidden_features,attention_layers,num_heads,linear_drop)

model_dict = model.state_dict()
model_path = r"logs\Exp1_SA.pth"
pretrained_dict = torch.load(model_path)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

root = r'experiment\exp1\test_rich'
out = r'results\exp1\SA\test_rich'
for file in os.listdir(root):
    filepath = os.path.join(root, file)
    file_data = pd.read_csv(filepath)
    datatestx = np.array(file_data[index]).astype(float)
    datatesty = np.array(file_data[["ET"]]).astype(float)
    for i in range(datatestx.shape[1]):
        input_data = datatestx[:, i:i + 1]
        input_data = (input_data - minx[i]) / (maxx[i] - minx[i])
        datatestx[:, i:i + 1] = input_data
    datapre = torch.FloatTensor(datatestx)
    results = []
    for i in range(datatestx.shape[0]):
        with torch.no_grad():
            output = model(datapre[i:i + 1, :])
            output = torch.squeeze(output, 0).numpy().tolist()
            results.append(output)
    results = np.array(results)
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
