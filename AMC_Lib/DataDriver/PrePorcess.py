from numpy import *
from scipy.io import loadmat
import os


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
    output    = zeros([2,n_ch,l_out])
    buff_poly = zeros([n_ch,l_fil])
    buff_cplx = zeros(n_ch)
    # filtered
    for i in range(l_fil):
        buff_poly = input_poly[:,i:i+l_fil]
        buff_cplx = fft.fftshift(n_ch*fft.ifft(sum(buff_poly*filter,axis=1)))
        output[0,:,i] = real(buff_cplx)
        output[1,:,i] = imag(buff_cplx)
    # return
    return output

def FilterBank32(input):
    PATH_Filter = os.path.dirname(__file__) + '/coeff/filter_poly.mat'
    filter_mat = loadmat(PATH_Filter)
    filter_poly = filter_mat['filter_poly']
    return FilterBank(input,filter_poly)


if __name__ == '__main__':
    input_IQ  = ones([1024,2])
    output = FilterBank32(input_IQ)
    print(output)
    print(output.shape)  