import numpy as np
import torch

def Accuracy(pred_labels,true_labels) -> float: 
    # validate input shapes
    if pred_labels.shape != true_labels.shape:
        raise ValueError("The size of predict-labels and true-labels should be the same.")
    # Main Proecess
    num_preds = pred_labels.shape[0]
    acc = (pred_labels==true_labels).sum() / num_preds
    return acc

def ConfMatrix(pred_labels,true_labels,norm=False):
    # validate input shapes
    if pred_labels.shape != true_labels.shape:
        raise ValueError("The size of predict-labels and true-labels should be the same.")
    # Main Proecess
    num_preds = pred_labels.shape[0]
    num_classes = max(true_labels)+1
    conf_matrix = np.zeros([num_classes,num_classes])
    for i in range(num_preds):
        conf_matrix[int(true_labels[i]),int(pred_labels[i])] += 1
    # Normalize
    if norm:
        for i in range(num_classes):
            conf_matrix[i, :] = conf_matrix[i, :] / np.sum(conf_matrix[i, :])
    # Return
    return conf_matrix

def Evaluator(dataloader,net):
    pred_labels = np.array([])
    true_labels = np.array([])
    for step, (signals, labels) in enumerate(dataloader,1):
        with torch.no_grad():
            preds = net(signals)
        labels = labels.cpu()
        preds = preds.cpu()
        pred_labels = np.hstack((pred_labels, preds.argmax(dim=-1).numpy()))
        true_labels = np.hstack((true_labels, labels.numpy()))
    return pred_labels,true_labels

