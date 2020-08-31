'''
Author: your name
Date: 2020-08-27 14:34:30
LastEditTime: 2020-08-27 14:42:03
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /AutoGeneration/dying_relu/train_34.py
'''
import os
import argparse
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.setrecursionlimit(1000000)
import uuid
sys.path.append('../../../data')
sys.path.append('../../../utils')
from utils_repair import *
from modules_repair import *
import numpy as np
import keras
import keras.optimizers as O
import argparse
import pickle
import itertools
import importlib
import keras.backend as K

from keras.datasets import mnist
from keras.models import load_model
from keras import backend as K
import pandas
import keras
import sys
def minist_load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = minist_load_data()
epoch = 50
train_num = 100
opt = 'adam'
dataset={}
dataset['x']=x_train
dataset['x_val']=x_test
dataset['y']=y_train
dataset['y_val']=y_test
batch_size=32
callbacks=[]
log_dir='./tmp_log_dir'
save_dir='./tmp_save_dir'
root_path='/data/zxy/DL_tools/DL_tools/code/AutoGeneration/'
new_issue_dir='/data/zxy/DL_tools/DL_tools/code/AutoGeneration/tmp_new_issue'
loss='categorical_crossentropy'
training_config={'batchsize': 32, 'dataset': 'mnist', 'epoch': 50, 'loss': 'categorical_crossentropy', 'opt_kwargs': {'lr': 0.001}, 'optimizer': 'Adam'}


for i in range(1):
    model = load_model("minist_relu_layer=34.h5")
    train_result,_,_=model_train(model=model,train_config_set=training_config,optimizer=opt,loss=loss,dataset=dataset,iters=epoch,batch_size=batch_size,\
                log_dir=log_dir,callbacks=callbacks,verb=1,checktype='epoch_3',autorepair=True,save_dir=save_dir,monitor=True,determine_threshold=5,\
                root_path=root_path,new_issue_dir=new_issue_dir)
