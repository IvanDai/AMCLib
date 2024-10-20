from numpy import *
import os
import h5py
import pickle

class splitRML2018():
    """
    A module desigend for the split of rml2018 with the indices as the output.
    [Note]:
    We can use this code to save the index directly:
    > splitRML2018(DATA_PATH).save(INDX_PATH)
    And can use this code to get the index variants:
    > train_idx, val_idx, test_idx = splitRML2018(DATA_PATH).get_index()
    """
    def __init__(self,DATA_PATH,split=[0.6,0.2,0.2]):
        """
        [Parameters]:
        DATA_PATH   - PATH of the dataset.
        spilt  - filter input with shape of [n_ch,l_fil], n_ch is the channel num.
        [Returns]:
        output  - signal output with the shape of [n_ch,l_sig/n_ch].
        """
        # load Dataset
        DATASET = h5py.File(DATA_PATH,'r')
        # get element numbers in train/validate/test sets
        n_examples = DATASET['X'].shape[0]
        n_train = int(n_examples * split[0])
        n_val   = int(n_examples * split[1])
        # generate index for each set
        self.train_idx = list(random.choice(range(0, n_examples), size=n_train, replace=False))
        self.val_idx   = list(random.choice(list(set(range(0,n_examples))-set(self.train_idx)), size=n_val, replace=False))
        self.test_idx  = list(set(range(0, n_examples)) - set(self.train_idx)-set(self.val_idx))

    def save(self,INDX_PATH):
        self_list = {
            "train" :self.train_idx,
            "val"   :self.val_idx,
            "test"  :self.test_idx
            }
        with open(INDX_PATH+'RML2018_idx.pkl', 'wb') as f:
            pickle.dump(self_list,f)
            f.close()
        return
    
    def get_index(self):
        return self.train_idx, self.val_idx, self.test_idx

if __name__ == '__main__':
    # get path
    CURR_PATH = os.path.abspath(os.path.dirname(__file__))
    ROOT_PATH = CURR_PATH[:CURR_PATH.find('AMC_Lib/')+len('AMC_Lib/')]
    DATA_PATH = ROOT_PATH + 'Datasets/RML2018.hdf5'
    INDX_PATH = ROOT_PATH + 'Source/Saves/Index/'
    # save index
    splitRML2018(DATA_PATH).save(INDX_PATH)