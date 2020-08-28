import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence

def load_data(max_features=5000):
    (x, y), (x_val,y_val) = imdb.load_data(num_words=max_features)
    return (x, y), (x_val, y_val)

def preprocess(x,maxlen=500,bk='tensorflow'):
    x_test = np.copy(x)
    if bk in ['tensorflow', 'cntk', 'theano']:
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    return x_test
