from tensorflow import keras
import scipy.io as io
from numpy import *
from numpy import floor
import h5py
import os

from PrePorcess import FilterBank


CURR_PATH = os.path.abspath(os.path.dirname(__file__))
ROOT_PATH = CURR_PATH[:CURR_PATH.find('AMC_Lib/')+len('AMC_Lib/')]
FILTER_PATH = CURR_PATH + 'Source/Saves/Others/filter_poly.mat'


class gen_RML2018_FB(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, DATA_PATH, indices, FILTER_PATH=FILTER_PATH, batch_size=32, dim=(1024,2),
                 classes=24, shuffle=True):
        # Initialization
        self.batch_size = batch_size    # batch
        self.indices    = indices       # train_idx 
        self.shuffle    = shuffle       # true
        self.l_sig      = dim[0]        # 1024
        self.n_IQ       = dim[1]        # 2 (I/Q)
        self.classes    = classes       # 24
        self.type       = type
        # get the DATASET
        self.DATASET    = h5py.File(DATA_PATH)
        self.filter     = io.loadmat(FILTER_PATH)['filter_poly']
        self.n_ch       = self.filter.shape[0]
        self.l_fil      = self.filter.shape[1]
        # Generator batch
        self.__on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(floor(len(self.indices)/self.batch_size))
    
    def __on_epoch_end(self):
        'Updates indices after each epoch'
        self.iindices = arange(len(self.indices))   # self.iindices is the order of self.indices
        if self.shuffle == True:
            random.shuffle(self.iindices)
    
    def __getitem__(self, start_idx):
        'Generate one batch of data'
        # Get the order of the data indices generated in this batch
        batch_iindices = self.iindices[ start_idx*self.batch_size : (start_idx+1)*self.batch_size]
        # Find list of IDs
        batch_indices = [self.indices[i] for i in batch_iindices]
        # Generate data
        X, Y = self.__data_generation(batch_indices)
        return X, Y

    def __data_generation(self, batch_indices):
        'Generates data containing batch_size samples'
        # Initialization
        X = empty((self.batch_size, self.n_ch, self.l_sig//self.n_ch, self.n_IQ)) # [none,32,32,2]
        Y = empty((self.batch_size, self.classes))          # [none,24]
        X_temp = empty((self.l_sig, self.n_IQ))             # [1024,2]
        # Generate data
        for i, idx in enumerate(batch_indices):
            # Store sample
            X_temp = self.DATASET['X'][idx,0:self.l_sig]    # [1024,2] IQ
            X[i,]  = FilterBank(X_temp,self.filter)      # aim at IQ signals
            Y[i]   = self.DATASET['Y'][idx]        
        # X: [none,32,32,2] Y:[none,24]
        return X,Y                      # batch*1024*2



"""
[Discription]:
    Standard generator for RML2018.

"""
class gen_RML2018(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, DATA_PATH, indices, batch_size=32, dim=(1024,2),
                 classes=24, shuffle=True, type='1024x2'):
        # Initialization
        self.batch_size = batch_size    # batch
        self.indices    = indices       # train_idx 
        self.shuffle    = shuffle       # true
        self.l_sig      = dim[0]        # 1024
        self.n_IQ       = dim[1]        # 2 (I/Q)
        self.classes    = classes       # 24
        self.type       = type
        # get the DATASET
        self.DATASET    = h5py.File(DATA_PATH)
        # Generator batch
        self.__on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(floor(len(self.indices)/self.batch_size))
    
    def __on_epoch_end(self):
        'Updates indices after each epoch'
        self.iindices = arange(len(self.indices))   # self.iindices is the order of self.indices
        if self.shuffle == True:
            random.shuffle(self.iindices)
    
    def __getitem__(self, start_idx):
        'Generate one batch of data'
        # Get the order of the data indices generated in this batch
        batch_iindices = self.iindices[ start_idx*self.batch_size : (start_idx+1)*self.batch_size]
        # Find list of IDs
        batch_indices = [self.indices[i] for i in batch_iindices]
        # Generate data
        X, Y = self.__data_generation(batch_indices)
        return X, Y

    def __data_generation(self, batch_indices):
        'Generates data containing batch_size samples'
        # Initialization
        X = empty((self.batch_size, self.l_sig, self.n_IQ)) # [none,1024,2]
        Y = empty((self.batch_size, self.classes))          # [none,24]
        # Generate data
        for i, idx in enumerate(batch_indices):
            # Store sample
            X[i,] = self.DATASET['X'][idx,0:self.l_sig]       # aim at IQ signals
            Y[i]  = self.DATASET['Y'][idx]
        if self.type == '1024x2' :          
            return X,Y                      # batch*1024*2
        elif self.type == '2x1024':
            return X.transpose(0,2,1), Y    # batch*2*1024