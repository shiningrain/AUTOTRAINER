import os
import sys
sys.path.append('../../data')
sys.path.append('../../utils')
from cifar10 import load_data,preprocess
from utils import *
import numpy as np
import keras
from keras.optimizers import SGD,Adam
import datetime
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,InputLayer
from matplotlib import pyplot
from keras.datasets import cifar10
labels=10
stringx='dying_relu_cnn_1'

(x, y), (x_val, y_val)=load_data()
x=preprocess(x,'tensorflow')
x_val=preprocess(x_val,'tensorflow')
y = keras.utils.to_categorical(y, labels)
y_val = keras.utils.to_categorical(y_val, labels)

x=x[:10000,:,:,:]
y=y[:10000,:]
x_val=x_val[:2000,:,:,:]
y_val=y_val[:2000,:]

#opt='SGD'
#opt=SGD(lr=0.01)
opt='Adam'
loss='categorical_crossentropy'
dataset={}
dataset['x']=x
dataset['y']=y
dataset['x_val']=x_val
dataset['y_val']=y_val
epoch=30
batch_size=256
log_dir='../../log/case14_cnn_tmp/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path=log_dir+'case14_cnn_tmp.csv'
fig_name=log_dir+'case14_cnn_tmp.pdf'
callbacks=[]

init ='RandomUniform'#'he_uniform'#keras.initializers.RandomUniform(minval=-1,maxval=1,seed=None)#
model=Sequential()
model.add(InputLayer(input_shape = (32, 32, 3)))
model.add(Conv2D(filters = 16,
          kernel_size = (3, 3),
          padding = 'same',       
          activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters = 36,
                kernel_size = (3, 3),
                padding = 'same',
                activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
'''
model.add(Dense(128, activation = 'selu',kernel_initializer='lecun_normal',bias_initializer='lecun_normal'))
model.add(Dense(128, activation = 'selu',kernel_initializer='lecun_normal',bias_initializer='lecun_normal'))
model.add(Dense(128, activation = 'selu',kernel_initializer='lecun_normal',bias_initializer='lecun_normal'))
model.add(Dense(128, activation = 'selu',kernel_initializer='lecun_normal',bias_initializer='lecun_normal'))
model.add(Dense(128, activation = 'selu',kernel_initializer='lecun_normal',bias_initializer='lecun_normal'))
model.add(Dense(128, activation = 'selu',kernel_initializer='lecun_normal',bias_initializer='lecun_normal'))
model.add(Dense(128, activation = 'selu',kernel_initializer='lecun_normal',bias_initializer='lecun_normal'))
model.add(Dense(128, activation = 'selu',kernel_initializer='lecun_normal',bias_initializer='lecun_normal'))
model.add(Dense(128, activation = 'selu',kernel_initializer='lecun_normal',bias_initializer='lecun_normal'))
model.add(Dense(128, activation = 'selu',kernel_initializer='lecun_normal',bias_initializer='lecun_normal'))
model.add(Dense(128, activation = 'selu',kernel_initializer='lecun_normal',bias_initializer='lecun_normal'))
model.add(Dense(128, activation = 'selu',kernel_initializer='lecun_normal',bias_initializer='lecun_normal'))
'''
model.add(Dense(128, activation = 'relu',kernel_initializer=init,bias_initializer='zeros'))
model.add(Dense(128, activation = 'relu',kernel_initializer=init,bias_initializer='zeros'))
model.add(Dense(128, activation = 'relu',kernel_initializer=init,bias_initializer='zeros'))
model.add(Dense(128, activation = 'relu',kernel_initializer=init,bias_initializer='zeros'))
model.add(Dense(128, activation = 'relu',kernel_initializer=init,bias_initializer='zeros'))
model.add(Dense(128, activation = 'relu',kernel_initializer=init,bias_initializer='zeros'))
model.add(Dense(128, activation = 'relu',kernel_initializer=init,bias_initializer='zeros'))
model.add(Dense(128, activation = 'relu',kernel_initializer=init,bias_initializer='zeros'))
model.add(Dense(128, activation = 'relu',kernel_initializer=init,bias_initializer='zeros'))
model.add(Dense(128, activation = 'relu',kernel_initializer=init,bias_initializer='zeros'))
model.add(Dense(128, activation = 'relu',kernel_initializer=init,bias_initializer='zeros'))
model.add(Dense(10, activation = 'softmax'))

print(model.summary())
save_model(model,'/data/zxy/DL_tools/DL_tools/models/DyingReLU/case14_cnn_tmp.h5')
model,history=model_train(model=model,optimizer=opt,loss=loss,dataset=dataset,iters=epoch,batch_size=batch_size,log_path=log_path,callbacks=callbacks,verb=1)
gradient_list,layer_outputs,wts=gradient_test(model,dataset,batch_size)
tmp_list=[]
for i in range(len(gradient_list)):
    zeros=np.sum(gradient_list[i]==0)
    tmp_list.append(zeros/gradient_list[i].size)
ave_g,_=average_gradient(gradient_list)
issue_list=determine_issue(gradient_list,history,layer_outputs,model,threshold_low=1e-3,threshold_high=1e+3)
#problem=gradient_issue(gradient_list)
result_dic=read_csv(log_path,epoch)
generate_fig(result_dic,fig_name)
print('finish')