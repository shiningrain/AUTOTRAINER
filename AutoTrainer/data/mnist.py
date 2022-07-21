import numpy as np
from tensorflow.keras.datasets import mnist

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
    
    
    # x_test /= 255
    # def preprocess(data_array):
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    # X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    x_test = scaler.fit_transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
    return x_test


if __name__=="__main__":
    (x, y), (x_val, y_val)=load_data()
    x1=preprocess(x)
    x2=preprocess_1(x)
    print(1)