import numpy as np
from tensorflow.keras.datasets import cifar10

def load_data():
    (x, y), (x_val, y_val) = cifar10.load_data()
    x=x.dot(100)
    x_val=x_val.dot(100)
    y = y.ravel()
    y_val = y_val.ravel()
    return (x, y), (x_val, y_val)

def preprocess(x, bk='tensorflow'):
    x_test = x.copy()/100
    if bk in ['tensorflow', 'cntk', 'theano']:
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    elif bk in ['mxnet', 'pytorch']:
        x_test = x_test.transpose((0, 3, 1, 2))
        x_test = x_test.reshape(x_test.shape[0], 3, 32, 32)
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_test[:, i, :, :] = (x_test[:, i, :, :] - mean[i]) / std[i]
    return  x_test