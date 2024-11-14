import torch

from torchkeras import summary,KerasModel

from datetime import datetime
import os
import pickle as pkl
import pandas as pd

from DataDriver.DataLoader import DS_RML2018
from DataDriver.PrePorcess import FilterBank32
from Models.ResNet import ResNet,BasicBlock

from Utils import Accuracy
from Utils.DictMaps import cri_dict,optim_dict

# =========== Configs =================================
configs = {
    # == Basic Settings ==
    'name_ds'   : "RML2018" ,
    'name_mod'  : "ResNet"  ,
    'name_tf'   : "FB"      ,
    'run_id'    : 1         ,

    # == Preporcess threads ==
    'num_workers' : 8       ,

    # == Datapath ==
    # 'DATA_PATH' : '../Datasets/RML2018.hdf5'

    # == Compile Settings ==
    'optimizer' : 'Adam'        ,
    'criterion' : 'CrossEntropy',
    'metrics'   : {"acc":Accuracy()},
    'learn_rate': 5e-3          ,

    # == Fit Settings ==
    'epochs'      : 2           ,
    'batch_size'  : 1024        ,
    'patience'    : 10          ,

    # == Training Mode ==
    'load_weight' : True        ,
    'load_log'    : True
}

# # == PATH Settings ==
# ROOT_PATH = os.path.abspath('.')[:os.path.abspath('.').find('AMC_Lib')+len('AMC_Lib')]
# DATA_PATH = ROOT_PATH + '/Datasets/RML2018.hdf5'
# # == Compile Settings ==
# # optimizer = None
# # criterion = None
# # metrics = None
# lr = 5e-3
# # == Fit Settings ==
# epochs   = 2
# batch_size  = 1024
# patience = 10


if __name__ == '__main__' :
    # =========== Prepare =================================
    # Name saved
    run_id     = configs['run_id'] if ('run_id' in configs) else datetime.now().strftime(r'%m%d_%H%M%S')
    name_ds    = configs['name_ds']
    name_mod   = configs['name_mod']
    name_saved = name_ds + '_' + name_mod + '_' + str(run_id)
    # Index Path
    ROOT_PATH = os.path.abspath('.')[:os.path.abspath('.').find('AMC_Lib')+len('AMC_Lib')]
    INDX_PATH = ROOT_PATH + '/Saves/Index/RML2018_idx.pkl'
    # Save Path
    SAVE_PATH_CP  = SAVE_PATH = ROOT_PATH+'/Saves/Checkpoints/'+name_saved+'.pt'
    SAVE_PATH_LOG = SAVE_PATH = ROOT_PATH+'/Saves/Checkpoints/'+name_saved+'.orc'
    # Data Path
    DATA_PATH = configs['DATA_PATH'] if ('DATA_PATH' in configs) else ROOT_PATH + '/Datasets/RML2018.hdf5'
    # =========== Data Loader =================================
    # == Load Configs
    batch_size  = configs['batch_size']
    num_workers = configs['num_workers']
    # == Get indices already splited
    with open(INDX_PATH,'rb') as f:
        all_indices = pkl.load(f)
    train_idx = all_indices["train"]
    valid_idx = all_indices["val"]
    # == Get Datasets 
    ds_train = DS_RML2018(DATA_PATH,train_idx,transform=FilterBank32)
    ds_valid = DS_RML2018(DATA_PATH,valid_idx,transform=FilterBank32)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # =========== Build Model =================================
    # == Load Configs
    loss_fn = cri_dict[configs['criterion']]
    optim   = optim_dict[configs['optimizer']]
    lr      = configs['learn_rate']
    epochs  = configs['epochs']
    patience = configs['patience']
    # == Model Builded 
    net = ResNet(BasicBlock,[2,2,2,2],24)
    summary(net,input_shape=(2,32,32))
    # == Model Compile ==
    model = KerasModel(net,
                    loss_fn   = loss_fn(),
                    optimizer = optim(net.parameters(),lr=lr),
                    metrics_dict = {"acc":Accuracy()}
                    )
    # =========== Load Weights & Train =================================
    if configs['load_weight']:
        model.load_ckpt(SAVE_PATH_CP)
    # == Model Train ==
    history = model.fit(train_data = dl_train, 
                        val_data   = dl_valid, 
                        epochs     = epochs, 
                        patience   = patience, 
                        monitor    = "val_acc",
                        mode       = "max",
                        ckpt_path  = SAVE_PATH_CP,
                        plot       = True
                        )
    # == Save Logs
    if configs['load_log']:
        log = pd.read_orc(SAVE_PATH_LOG)
        log = pd.concat([log,history],ignore_index=True)
        log['epoch'] = list(range(1,len(log)+1))
    else:
        log = history
    log.to_orc(SAVE_PATH_LOG)