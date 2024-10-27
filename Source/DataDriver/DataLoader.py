import os
from torch.utils.data import Dataset, DataLoader
import h5py
import pickle as pkl
import torch
# from torchvision.transforms import transforms

class RML2018_Dataset(Dataset):
    def __init__(self, DATA_PATH, indices, transform=None):
        self.indices    = indices
        self.DATASET    = h5py.File(DATA_PATH)
        self.transform  = transform

    def __getitem__(self, iindex):
        index = self.indices[iindex]
        signal = self.DATASET['X'][index,:,:]
        label  = self.DATASET['Y'][index,:]
        if self.transform is not None:
            signal = self.transform(signal)
        signal = torch.Tensor(signal)
        label = int(torch.argmax(torch.Tensor(label)))
        return signal,label
    
    def __len__(self):
        return len(self.indices)
 

if __name__ == '__main__':
    # Get PATHs
    CURR_PATH = os.path.abspath(os.path.dirname(__file__))
    ROOT_PATH = CURR_PATH[:CURR_PATH.find('AMC_Lib')+len('AMC_Lib/')]
    DATA_PATH = ROOT_PATH + 'Datasets/RML2018.hdf5'
    IDX_PATH  = ROOT_PATH + 'Saves/Index/RML2018_idx.pkl'
    # Get indices already splited
    with open(IDX_PATH,'rb') as f:
        all_indices = pkl.load(f)
    train_idx = all_indices["train"]
    # Test DataSet
    from PrePorcess import FilterBank32
    train_ds = RML2018_Dataset(DATA_PATH,train_idx,FilterBank32)
    train_dl = DataLoader(train_ds,32)
    for data,label in train_dl:
        print(label)
        break