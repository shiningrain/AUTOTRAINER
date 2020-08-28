'''
@Author: your name
@Date: 2020-07-21 06:50:46
@LastEditTime: 2020-07-21 06:57:00
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /GradientVanish/data/zxy/DL_tools/DL_tools/code/demo_vanish/demo.py
'''
import os
from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,BatchNormalization,Dropout,Input
from keras.models import Sequential,Model
from keras.optimizers import SGD,Adam
from keras.initializers import RandomUniform
from matplotlib import pyplot
import sys
sys.path.append('../../data')
sys.path.append('../../utils')
sys.path.append('../../modules')
from simplednn import load_data,preprocess
from utils import *
from modules import *
import numpy as np
import keras

(x, y), (x_val, y_val)=load_data()
opt='Adam'#'SGD'#
loss='binary_crossentropy'
dataset={}
dataset['x']=x
dataset['y']=y
dataset['x_val']=x_val
dataset['y_val']=y_val
epoch=100
batch_size=128
log_dir='/data/zxy/DL_tools/DL_tools/code/demo_vanish/train_log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path=log_dir+'log.csv'
fig_name=log_dir+'log.pdf'
callbacks=[]

config={}
config['opt']=opt
config['loss']=loss
config['dataset']=dataset
config['epoch']=epoch
config['batch_size']=batch_size

model_path='/data/zxy/DL_tools/DL_tools/code/demo_vanish/simplednn.h5'
model=load_model(model_path)
print(model.summary())
train_result,trained_model,model_path=model_train(model=model,optimizer=opt,loss=loss,dataset=dataset,iters=epoch,batch_size=batch_size,\
    log_path=log_path,callbacks=callbacks,verb=1,checktype='epoch_3',autorepair=True)
print('finish')
