import torch
import torch.nn as nn
from torch import Tensor
# from torchkeras import summary,KerasModel

import os
import pickle as pkl
from sklearn.metrics import accuracy_score

from DataDriver.DataLoader import RML2018_Dataset
from DataDriver.PrePorcess import FilterBank32
from Models.ResNet import ResNet,BasicBlock

batch_size  = 512
num_workers = 8
name_ds  = "RML2018"
name_mod = "ResNet"
name_tf  = "FB"

# Get PATHs
# CURR_PATH = os.path.abspath(os.path.dirname(__file__))
CURR_PATH = os.path.abspath('.')
ROOT_PATH = CURR_PATH[:CURR_PATH.find('AMC_Lib')+len('AMC_Lib/')]
DATA_PATH = ROOT_PATH + 'Datasets/RML2018.hdf5'
IDX_PATH  = ROOT_PATH + 'Saves/Index/RML2018_idx.pkl'
# Get indices already splited
with open(IDX_PATH,'rb') as f:
    all_indices = pkl.load(f)
train_idx = all_indices["train"]
valid_idx = all_indices["val"]
# Get Datasets
ds_train = RML2018_Dataset(DATA_PATH,train_idx,transform=FilterBank32)
ds_valid = RML2018_Dataset(DATA_PATH,valid_idx,transform=FilterBank32)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

net = ResNet(BasicBlock,[2,2,2,2],24)
# summary(net,input_shape=(2,32,32))


# =========================================
import datetime
import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score
# from torchmetrics.classification import Accuracy

def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true.cpu().numpy(),y_pred_cls.cpu().numpy())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = net.to(device) # move to gpu

model.optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
model.loss_func = nn.CrossEntropyLoss()
model.metric_func = accuracy
model.metric_name = "accuracy"

def train_step(model,features,labels,device='cpu'):

    # 训练模式，dropout层发生作用
    model.train()
    features = features.to(device)
    labels = labels.to(device)

    # 梯度清零
    model.optimizer.zero_grad()

    # 正向传播求损失
    predictions = model(features)
    loss = model.loss_func(predictions,labels)
    metric = model.metric_func(predictions,labels)

    # 反向传播求梯度
    loss.backward()
    model.optimizer.step()
    # print(loss)
    # print(metric)
    # return loss.item(),metric.item()
    return loss.item(),metric

@torch.no_grad()
def valid_step(model,features,labels,device='cpu'):

    # 预测模式，dropout层不发生作用
    model.eval()

    features = features.to(device)
    labels = labels.to(device)

    predictions = model(features)
    loss = model.loss_func(predictions,labels)
    metric = model.metric_func(predictions,labels)

    # return loss.item(), metric.item()
    return loss.item(), metric


# 测试train_step效果
features,labels = next(iter(dl_train))
features = features.to(device)
labels   = labels.to(device)
train_step(model,features,labels,device)

def train_model(model,epochs,dl_train,dl_valid,log_step_freq,device='cpu'):

    metric_name = model.metric_name
    dfhistory = pd.DataFrame(columns = ["epoch","loss",metric_name,"val_loss","val_"+metric_name]) 
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=========="*8 + "%s"%nowtime)

    for epoch in range(1,epochs+1):  

        # 1，训练循环-------------------------------------------------
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, (features,labels) in enumerate(dl_train, 1):

            loss,metric = train_step(model,features,labels,device)

            # 打印batch级别日志
            loss_sum += loss
            metric_sum += metric
            if step%log_step_freq == 0:   
                print(("[step = %d] loss: %.3f, "+metric_name+": %.3f") %
                      (step, loss_sum/step, metric_sum/step))

        # 2，验证循环-------------------------------------------------
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features,labels) in enumerate(dl_valid, 1):

            val_loss,val_metric = valid_step(model,features,labels,device)

            val_loss_sum += val_loss
            val_metric_sum += val_metric

        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum/step, metric_sum/step, 
                val_loss_sum/val_step, val_metric_sum/val_step)
        dfhistory.loc[epoch-1] = info

        # 打印epoch级别日志
        print(("\nEPOCH = %d, loss = %.3f,"+ metric_name + \
              "  = %.3f, val_loss = %.3f, "+"val_"+ metric_name+" = %.3f") 
              %info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"=========="*8 + "%s"%nowtime)

    print('Finished Training...')
    return dfhistory


epochs = 50
dfhistory = train_model(model,epochs,dl_train,dl_valid,log_step_freq = 500,device=device)