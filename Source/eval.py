import torch

from torchkeras import summary

from datetime import datetime
import os
import pickle as pkl
import numpy as np

from DataDriver.DataLoader import DS_RML2018
from DataDriver.PrePorcess import FilterBank32
from Models.ResNet import ResNet,BasicBlock

from Utils import Accuracy,plot_confusion_matrix
from accelerate import Accelerator

# =========== Configs =================================
configs = {
    # == Basic Settings ==
    'name_ds'   : "RML2018" ,
    'name_mod'  : "ResNet"  ,
    'name_tf'   : "FB"      ,
    'run_id'    : 1         ,
    'n_classes' : 24        ,

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

# =========== Class Map =================================
class_map = ['OOK',  '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', 
             '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', 
             '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 
             'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']

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
    test_idx = all_indices["test"]
    valid_idx = all_indices["val"]

    # =========== Device =================================
    accelerator = Accelerator()       # to GPU
    accelerator.print(f'device {str(accelerator.device)} is used!')

    # =========== Model =================================
    num_classes = configs['n_classes']
    net = ResNet(BasicBlock,[2,2,2,2],num_classes=num_classes)
    summary(net,input_shape=(2,32,32))
    net.load_state_dict(torch.load(SAVE_PATH_CP))

    # =========== Test: inference speed =================================
    ds_test = DS_RML2018(DATA_PATH,test_idx,transform=FilterBank32)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # == GPU
    net,dl_test = accelerator.prepare(net,dl_test)
    # == variants definination
    accurate_preds = 0
    num_preds = 0
    predictions = np.array([])
    true_labels = np.array([])
    # >>> start evaluation
    import time
    start = time.perf_counter()
    print(f"Start Prediction")
    for step, (signals, labels) in enumerate(dl_test,1):
        with torch.no_grad():
            preds = net(signals)
        labels = labels.cpu()
        preds = preds.cpu()
        predictions = np.hstack((predictions, preds.argmax(dim=-1).numpy()))
        true_labels = np.hstack((true_labels, labels.numpy()))
    # <<< end evaluation
    print(f">>>>>> Predict Time: {time.perf_counter() - start}s <<<<<<")
    # == overall accuracy
    num_preds = predictions.shape[0]
    accurate_preds += (predictions==true_labels).sum() / num_preds
    print(f" | \n |--> Overall Acc : {accurate_preds*100}% ")
    # == confusion matrix
    conf_matrix = np.zeros([num_classes,num_classes])
    for i in range(num_preds):
        conf_matrix[int(true_labels[i]),int(predictions[i])] += 1
    for i in range(num_classes):
        conf_matrix[i, :] = conf_matrix[i, :] / np.sum(conf_matrix[i, :])
    plot_confusion_matrix(conf_matrix, labels=class_map, title='%s Confusion matrix' % (name_saved))

    




    
