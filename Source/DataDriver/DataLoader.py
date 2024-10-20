import os
from torch.utils.data import Dataset, DataLoader
import h5py
import pickle as pkl
from PrePorcess import FilterBank32
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
        return signal,label
    
    def __len__(self):
        return self.indices.shape[0]



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
    train_ds = RML2018_Dataset(DATA_PATH,train_idx,FilterBank32)
    # print(train_ds[1][0].shape)
