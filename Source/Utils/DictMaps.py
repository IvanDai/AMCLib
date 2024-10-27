import torch
import torch.nn as nn

cri_dict = {
    'CrossEntropy' : nn.CrossEntropyLoss
}

optim_dict = {
    'Adam' : torch.optim.Adam
}