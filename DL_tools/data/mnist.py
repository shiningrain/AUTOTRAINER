import numpy as np
from keras.datasets import mnist

def load_data():
    (x, y), (x_val, y_val) = mnist.load_data()
    return (x, y), (x_val, y_val)

def preprocess(x, bk='tensorflow'):
    x_test = np.copy(x)
    if bk in ['tensorflow', 'cntk', 'theano']:
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    elif bk in ['mxnet', 'pytorch']:
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    x_test = x_test.astype('float32')
    x_test /= 255
    return x_test
    #return x_test.reshape((-1,28,28))

def preprocess_1(x, bk='tensorflow'):
    x_test = np.copy(x)
    if bk in ['tensorflow', 'cntk', 'theano']:
        x_test = x_test.reshape(x_test.shape[0], 28, 28)
    elif bk in ['mxnet', 'pytorch']:
        x_test = x_test.reshape(x_test.shape[0], 28, 28)
    x_test = x_test.astype('float32')
    x_test /= 255
    return x_test
