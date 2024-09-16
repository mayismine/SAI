import torch
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from task.ET.model.SA import Attention
# from ET.MS_Rain_Attention import MSMT_LE
import math
import os
from sklearn.metrics import mean_squared_error


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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

train_ratio = 0.9
Batchsize = 128
learning_rate = 0.5
Init_epoch = 0
Freeze_epoch = 1000


path = r'experiment\exp1\train'
test_val = r'experiment\exp1\validation'


max_min_data = get_data(r'experiment\exp1\maxmin')
datamaxmin = np.array(max_min_data[['WS','Ta','RH','Press','SW_IN','LAI','SIF','acc_rain7','SM','Month','DEM','GRA','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT']]).astype(float) # rain
maxx,minx,_ = data_norm(datamaxmin)
print(maxx)
print('==========')
print(minx)
print('==========')
datamaxminy = np.array(max_min_data[["ET"]]).astype(float)
maxy,miny,_ = data_norm(datamaxminy)
print(maxy)
print('==========')
print(miny)
print('==========')

dataval = get_data(test_val)
# datarichx = np.array(datarich[["Rain","WS","Ta", "P", "RH", "SW_IN",'SM','LAI','SIF',"Month",'DEM','GRA','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT']]).astype(float) # rain
datavalx = np.array(dataval[['WS','Ta','RH','Press','SW_IN','LAI','SIF','acc_rain7','SM','Month','DEM','GRA','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT']]).astype(float) # rain
datavaly = np.array(dataval[["ET"]]).astype(float)
for i in range(datavalx.shape[1]):
    input_data = datavalx[:, i:i + 1]
    input_data = (input_data - minx[i]) / (maxx[i] - minx[i])
    datavalx[:, i:i + 1] = input_data

data = get_data(path)
# datatrainx = np.array(data[["Rain","WS","Ta", "P", "RH", "SW_IN",'SM','LAI','SIF',"Month",'DEM','GRA','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT']]).astype(float) # rain
datatrainx = np.array(data[['WS','Ta','RH','Press','SW_IN','LAI','SIF','acc_rain7','SM','Month','DEM','GRA','WSA','SAV','EBF','DBF','MF','ENF','WET','CRO','OSH','CSH','CLAY','SAND','SILT']]).astype(float) # rain
datatrainy = np.array(data[["ET"]]).astype(float)
for i in range(datatrainx.shape[1]):
    input_data = datatrainx[:, i:i + 1]
    if maxx[i] != minx[i]:
        input_data = (input_data - minx[i]) / (maxx[i] - minx[i])
    datatrainx[:, i:i + 1] = input_data
for i in range(datatrainy.shape[1]):
    input_data = datatrainy[:, i:i + 1]
    if maxy[i] != miny[i]:
        input_data = (input_data - miny[i]) / (maxy[i] - miny[i])
    datatrainy[:, i:i + 1] = input_data

num = datatrainx.shape[0]
print(num)


trainx_data = datatrainx
trainy_data = datatrainy
num_train = len(datatrainx)
train_x = torch.from_numpy(trainx_data).float()
train_y = torch.from_numpy(trainy_data).float()
validationx_data = torch.from_numpy(datavalx).float()
validationy_data = torch.from_numpy(datavaly).float()
train_torch_dataset = Data.TensorDataset(train_x,train_y)

train_loader = DataLoader(train_torch_dataset,batch_size=Batchsize,shuffle=True,num_workers=0)

in_features = 25
sparameters = 7
num_heads = 3
depth = 1
hidden_features= 48
parameter_features = 20
gate_num = 11
control_num = 11
gate_True = True

attention_layers = 10
linear_drop = 0.1

model = Attention(in_features,hidden_features,attention_layers,num_heads,linear_drop)
modeltrain = model.train()
loss = nn.MSELoss()
epoch_step = num_train // Batchsize

model_path      = r""
if model_path != '':
    print('Loading weights into state dict...')
    model_dict = modeltrain.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    modeltrain.load_state_dict(model_dict)

if epoch_step == 0:
    raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

warm_up_epochs = int(Freeze_epoch * 0.3)

optimizer = torch.optim.SGD(modeltrain.parameters(), lr = learning_rate)
warm_up_cosine_lr = lambda epoch: epoch/warm_up_epochs if epoch<= warm_up_epochs else 0.5*(math.cos((epoch-warm_up_epochs)/(Freeze_epoch-warm_up_epochs)*math.pi)+1)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,warm_up_cosine_lr)

for epoch in range(Freeze_epoch):
    train_loss = []
    for step, (x_train, y_train) in enumerate(train_loader):
        output_train = modeltrain(x_train)
        lossvalue = loss(output_train,y_train)
        optimizer.zero_grad()
        lossvalue.backward()
        optimizer.step()
        train_loss.append(lossvalue.item())
    lr_scheduler.step()
    train_loss = np.array(train_loss)
    train_loss = np.mean(train_loss)
    # print(get_lr(optimizer))
    with torch.no_grad():
        validation_output = modeltrain(validationx_data)
    val = math.sqrt(mean_squared_error((np.array(validation_output)[:, 0:1]) * (maxy[0] - miny[0]) + miny[0], validationy_data))
    torch.save(modeltrain.state_dict(), r'logs\SA_' + str(epoch + 1) + 'train_' + str(train_loss)+ 'val_' + str(val)  + '.pth')
