from tensorflow import keras
import scipy.io as io
from numpy import *
from numpy import floor
import h5py
import os


CURR_PATH = os.path.abspath(os.path.dirname(__file__))
ROOT_PATH = CURR_PATH[:CURR_PATH.find('AMC_Lib/')+len('AMC_Lib/')]
FILTER_PATH = CURR_PATH + 'Source/Saves/Others/filter_poly.mat'


def FilterBank(input,filter):
    """
    Function to filtered the input vector into channels.
    [Parameters]:
        input   - signal input with length of l_sig, which should be in IQ form with [l_sig,2].
        filter  - filter input with shape of [n_ch,l_fil], n_ch is the channel num.
    [Returns]:
        output  - signal output with the shape of [n_ch,l_sig/n_ch].
    """
    # get parameter values
    l_sig      = input.shape[0]
    n_ch,l_fil = filter.shape
    l_out      = l_sig//n_ch
    input_cplx = input[:,0] + 1j*input[:,1]
    # pre-transform
    vec_supply = zeros([n_ch,l_fil])         # zero-vector with the same shape of the filter
    input_poly = reshape(input_cplx,(n_ch,-1))
    input_poly = concatenate((vec_supply[:,0:l_fil//2], input_poly , vec_supply[:,0:(l_fil+1)//2]) ,axis=1)
    # declare variables
    output    = zeros([n_ch,l_out,2])
    buff_poly = zeros([n_ch,l_fil])
    buff_cplx = zeros(n_ch)
    # filtered
    for i in range(l_fil):
        buff_poly = input_poly[:,i:i+l_fil]
        buff_cplx = fft.fftshift(n_ch*fft.ifft(sum(buff_poly*filter,axis=1)))
        output[:,i,0] = real(buff_cplx)
        output[:,i,1] = imag(buff_cplx)
    # return
    return output


class gen_RML2018_FB(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, DATA_PATH, idxes, FILTER_PATH=FILTER_PATH, batch_size=32, dim=(1024,2),
                 classes=24, shuffle=True):
        # Initialization
        self.batch_size = batch_size    # batch
        self.idxes      = idxes         # train_idx 
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
        return int(floor(len(self.idxes)/self.batch_size))
    
    def __on_epoch_end(self):
        'Updates indexes after each epoch'
        self.iidxes = arange(len(self.idxes))   # self.iidxes is the order of self.idxes
        if self.shuffle == True:
            random.shuffle(self.iidxes)
    
    def __getitem__(self, start_idx):
        'Generate one batch of data'
        # Get the order of the data indexes generated in this batch
        batch_iidxes = self.iidxes[ start_idx*self.batch_size : (start_idx+1)*self.batch_size]
        # Find list of IDs
        batch_idxes = [self.idxes[i] for i in batch_iidxes]
        # Generate data
        X, Y = self.__data_generation(batch_idxes)
        return X, Y

    def __data_generation(self, batch_idxes):
        'Generates data containing batch_size samples'
        # Initialization
        X = empty((self.batch_size, self.n_ch, self.l_sig//self.n_ch, self.n_IQ)) # [none,32,32,2]
        Y = empty((self.batch_size, self.classes))          # [none,24]
        X_temp = empty((self.l_sig, self.n_IQ))             # [1024,2]
        # Generate data
        for i, idx in enumerate(batch_idxes):
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
    def __init__(self, DATA_PATH, idxes, batch_size=32, dim=(1024,2),
                 classes=24, shuffle=True, type='1024x2'):
        # Initialization
        self.batch_size = batch_size    # batch
        self.idxes      = idxes         # train_idx 
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
        return int(floor(len(self.idxes)/self.batch_size))
    
    def __on_epoch_end(self):
        'Updates indexes after each epoch'
        self.iidxes = arange(len(self.idxes))   # self.iidxes is the order of self.idxes
        if self.shuffle == True:
            random.shuffle(self.iidxes)
    
    def __getitem__(self, start_idx):
        'Generate one batch of data'
        # Get the order of the data indexes generated in this batch
        batch_iidxes = self.iidxes[ start_idx*self.batch_size : (start_idx+1)*self.batch_size]
        # Find list of IDs
        batch_idxes = [self.idxes[i] for i in batch_iidxes]
        # Generate data
        X, Y = self.__data_generation(batch_idxes)
        return X, Y

    def __data_generation(self, batch_idxes):
        'Generates data containing batch_size samples'
        # Initialization
        X = empty((self.batch_size, self.l_sig, self.n_IQ)) # [none,1024,2]
        Y = empty((self.batch_size, self.classes))          # [none,24]
        # Generate data
        for i, idx in enumerate(batch_idxes):
            # Store sample
            X[i,] = self.DATASET['X'][idx,0:self.l_sig]       # aim at IQ signals
            Y[i]  = self.DATASET['Y'][idx]
        if self.type == '1024x2' :          
            return X,Y                      # batch*1024*2
        elif self.type == '2x1024':
            return X.transpose(0,2,1), Y    # batch*2*1024