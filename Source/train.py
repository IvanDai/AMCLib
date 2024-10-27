import torch
import torch.nn as nn
from torchkeras import summary,KerasModel

from datetime import datetime
import os
import pickle as pkl
import pandas as pd

from DataDriver.DataLoader import RML2018_Dataset
from DataDriver.PrePorcess import FilterBank32
from Models.ResNet import ResNet,BasicBlock

from Utils import Accuracy


# =========== Configs =================================
# == Basic Settings ==
name_ds  = "RML2018"
name_mod = "ResNet"
name_tf  = "FB"
run_id   = 1
num_workers = 8
# == PATH Settings ==
ROOT_PATH = os.path.abspath('.')[:os.path.abspath('.').find('AMC_Lib')+len('AMC_Lib')]
DATA_PATH = ROOT_PATH + '/Datasets/RML2018.hdf5'
# == Compile Settings ==
# optimizer = None
# criterion = None
# metrics = None
lr = 5e-3
# == Fit Settings ==
epochs   = 2
batch_size  = 1024
patience = 10





# =========== Prepare =================================
# Index Path
INDX_PATH = ROOT_PATH + '/Saves/Index/RML2018_idx.pkl'
# Save Path
if run_id is None: # use timestamp as default run-id
    run_id = datetime.now().strftime(r'%m%d_%H%M%S')
name_saved = name_ds + '_' + name_mod + '_' + str(run_id)
SAVE_PATH_CP  = SAVE_PATH = ROOT_PATH+'/Saves/Checkpoints/'+name_saved+'.pt'
SAVE_PATH_LOG = SAVE_PATH = ROOT_PATH+'/Saves/Checkpoints/'+name_saved+'.orc'
# Get indices already splited
with open(INDX_PATH,'rb') as f:
    all_indices = pkl.load(f)
train_idx = all_indices["train"]
valid_idx = all_indices["val"]
# =========== Build Model =================================
# == Get Datasets ==
ds_train = RML2018_Dataset(DATA_PATH,train_idx,transform=FilterBank32)
ds_valid = RML2018_Dataset(DATA_PATH,valid_idx,transform=FilterBank32)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)
# == Model Builded ==
# net = create_net()
net = ResNet(BasicBlock,[2,2,2,2],24)
summary(net,input_shape=(2,32,32))
# == Model Compile ==
model = KerasModel(net,
                   loss_fn = nn.CrossEntropyLoss(),
                   optimizer= torch.optim.Adam(net.parameters(),lr=lr),
                   metrics_dict = {"acc":Accuracy()}
                   )
# == Model Train ==
history = model.fit(train_data=dl_train, 
                      val_data=dl_valid, 
                      epochs=epochs, 
                      patience=patience, 
                      monitor="val_acc",
                      mode="max",
                      ckpt_path=SAVE_PATH_CP,
                      plot=True
                     )
dfhistory = pd.DataFrame()
dfhistory.loc[len(dfhistory)] = history

dfhistory.to_orc(SAVE_PATH_LOG)