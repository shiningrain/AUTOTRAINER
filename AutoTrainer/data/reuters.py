'''
@Author: your name
@Date: 2020-07-28 20:50:22
@LastEditTime: 2020-07-28 21:02:49
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /AutoGeneration/data/zxy/DL_tools/DL_tools/data/reuters.py
'''
import numpy as np
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing import sequence

def load_data(max_features=10000):
    (x, y), (x_val,y_val) = reuters.load_data(num_words=max_features)
    return (x, y), (x_val, y_val)

def preprocess(x,maxlen=300,bk='tensorflow'):
    x_test = np.copy(x)
    if bk in ['tensorflow', 'cntk', 'theano']:
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    return x_test
