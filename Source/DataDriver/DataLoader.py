import os
from torch.utils.data import Dataset, DataLoader
import h5py
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
    