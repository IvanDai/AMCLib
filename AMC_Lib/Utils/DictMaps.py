import torch
import torch.nn as nn

'''
Several mapping dicts are stored in this file
'''

cri_dict = {
    'CrossEntropy' : nn.CrossEntropyLoss
}

optim_dict = {
    'Adam' : torch.optim.Adam
}