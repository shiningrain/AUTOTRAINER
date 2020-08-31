'''
@Author: your name
@Date: 2020-05-29 10:09:06
@LastEditTime: 2020-08-03 19:39:39
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /AutoGeneration/data/zxy/DL_tools/DL_tools/data/imdb.py
'''
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

# a,b=load_data()
# print(1)