'''
@Author: your name
@Date: 2020-05-29 10:09:04
LastEditTime: 2020-08-17 10:59:22
LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /AutoGeneration/data/zxy/DL_tools/DL_tools/data/simplednn.py
'''
import numpy as np
from sklearn.datasets import make_circles, make_blobs,make_regression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os 
import pickle

def load_data(method='circle'):
    if method=='circle':
        X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
        # scale input data to [-1,1]
        
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # X = scaler.fit_transform(X)
        
        # split into train and test
        X=X*100
        
        n_train = 700
        x, x_val= X[:n_train, :], X[n_train:, :]
        y, y_val = y[:n_train], y[n_train:]
        # with open('./data_circle.pkl', 'rb') as f:#input,bug type,params
        #     dataset = pickle.load(f)
        # x=dataset['x']
        # y=dataset['y']
        # x_val=dataset['x_val']
        # y_val=dataset['y_val']
    elif method=='blob':
        X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
        # one hot encode output variable
        X=X*100
        y = to_categorical(y)
        # split into train and test
        n_train = 700
        x, x_val= X[:n_train, :], X[n_train:, :]
        y, y_val = y[:n_train], y[n_train:]
        # with open('/data/zxy/DL_tools/DL_tools/data/data_blob.pkl', 'rb') as f:#input,bug type,params
        #     dataset = pickle.load(f)
        # x=dataset['x']
        # y=dataset['y']
        # x_val=dataset['x_val']
        # y_val=dataset['y_val']
    elif method=='reg':
        X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
        # split into train and test
        X=X*100
        n_train = 500
        x, x_val = X[:n_train, :], X[n_train:, :]
        y, y_val = y[:n_train], y[n_train:]
    elif method=='relu':
        x=np.random.uniform(-1,0,(200000,4))
        y=np.random.randint(2,size=200000)
        x, x_val, y, y_val = train_test_split(x, y, test_size=0.3, random_state=0)
    else:
        print('Not Support This Method')
        os._exit(0)

    return (x,y),(x_val,y_val)

def preprocess(x):
    return x/100