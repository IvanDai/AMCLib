import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

import os,sys
pkg_pth = os.path.abspath('.')[:os.path.abspath('.').find('AMC_Lib')+len('AMC_Lib')]+'/Source'
sys.path.append(pkg_pth)
from DataDriver.DataLoader import DS_RML2018

# ======== Predict ========
def Predict(net,dataloader):
    '''
    [Parameters]:
    net        -    <torch.nn.module> 
    dataloader -    <torch.utils.data.DataLoader> It should be instantiated DataLoader.
    '''
    # To gpu
    accelerator = Accelerator()
    net,dataloader = accelerator.prepare(net,dataloader)
    # Predict
    pred_labels = np.array([])
    true_labels = np.array([])
    for _, (signals, labels) in enumerate(dataloader,1):
        with torch.no_grad():
            preds = net(signals)
        labels = labels.cpu()
        preds = preds.cpu()
        pred_labels = np.hstack((pred_labels, preds.argmax(dim=-1).numpy()))
        true_labels = np.hstack((true_labels, labels.numpy()))
    return pred_labels,true_labels

# ======== Get Accuracy ========
def GetAccuracy(pred_labels,true_labels) -> float: 
    # validate input shapes
    if pred_labels.shape != true_labels.shape:
        raise ValueError("The size of predict-labels and true-labels should be the same.")
    # Main Proecess
    num_preds = pred_labels.shape[0]
    acc = (pred_labels==true_labels).sum() / num_preds
    return acc

# ======== Get Confusion Matrix ========
def GetConfMatrix(pred_labels,true_labels,num_classes=24,norm=False):
    # validate input shapes
    if pred_labels.shape != true_labels.shape:
        raise ValueError("The size of predict-labels and true-labels should be the same.")
    # Main Proecess
    num_preds = pred_labels.shape[0]
    # num_classes = max(true_labels)+1
    conf_matrix = np.zeros([num_classes,num_classes])
    for i in range(num_preds):
        conf_matrix[int(true_labels[i]),int(pred_labels[i])] += 1
    # Normalize
    if norm:
        conf_matrix = NormConfMatrix(conf_matrix)
    # Return
    return conf_matrix

def NormConfMatrix(conf_matrix:np.ndarray):
    if np.size(conf_matrix,0) != np.size(conf_matrix,1):
        raise ValueError('Confusion matrix should be a square matrix with the shape like: 10x10')
    num_classes = np.size(conf_matrix,0)
    for i in range(num_classes):
        conf_matrix[i, :] = conf_matrix[i, :] / np.sum(conf_matrix[i, :])
    return conf_matrix


# ======== Evaluate for Each SNR ========
snrs = np.arange(-20,32,2)
def EvaluateSNR(net,idx_list,snr_list,
            DATAPATH,
            DatasetFunc=DS_RML2018, tf=None, batch_size=256, num_workers=8,
            snrs=snrs,num_classes=24):
    cf_rec = np.zeros([len(snrs),num_classes,num_classes])
    for i in range(len(snrs)):
        snr = snrs[i]
        idx_snr = idx_list[np.where(snr_list==snr)]
        ds_eval = DatasetFunc(DATAPATH,idx_snr,transform=tf)
        dl_eval = DataLoader(ds_eval, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        pred_labels,true_labels = Predict(net,dl_eval)
        cf_matrix_tmp = GetConfMatrix(pred_labels,true_labels,norm=False)
        cf_rec[i] = cf_matrix_tmp
    return cf_rec
