# mlp for the two circles classification problem
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


#1.两环问题的简单模型，精度到0.8左右难以提升
# generate 2d classification dataset
(x, y), (x_val, y_val)=load_data()

'''对数据集排序
z=zip(y.tolist(),x.tolist())
Z=sorted(z,reverse=True)
y,x=zip(*Z)
y=np.array(y)
x=np.array(x)
'''

opt='Adam'#'SGD'#
loss='binary_crossentropy'
dataset={}
dataset['x']=x
dataset['y']=y
dataset['x_val']=x_val
dataset['y_val']=y_val
epoch=100
batch_size=128
log_dir='../../log/case16/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path=log_dir+'case16.csv'
fig_name=log_dir+'case16.pdf'
callbacks=[]

config={}
config['opt']=opt
config['loss']=loss
config['dataset']=dataset
config['epoch']=epoch
config['batch_size']=batch_size

#model_path='/data/zxy/DL_tools/DL_tools/models/GradientVanish/case16.h5'
model_path='/data/zxy/DL_tools/DL_tools/models/seed_model/simplednn.h5'
#model = Sequential()
#init = RandomUniform(minval=0, maxval=1)
init = 'he_uniform'
input = Input(shape=(2,))
x=Dense(5,activation='sigmoid', kernel_initializer=init)(input)
x=Dense(5, activation='sigmoid', kernel_initializer=init)(x)
x=Dense(5, activation='sigmoid', kernel_initializer=init)(x)
x=Dense(5, activation='sigmoid', kernel_initializer=init)(x)
x=Dense(5, activation='sigmoid', kernel_initializer=init)(x)
x=Dense(5, activation='sigmoid', kernel_initializer=init)(x)
x=Dense(5, activation='sigmoid', kernel_initializer=init)(x)

x=Dense(1,activation='sigmoid', kernel_initializer=init)(x)
model=Model(input, x)
#model.add(Dense(5, input_dim=2, activation='sigmoid', kernel_initializer=init))
'''
model.add(Dense(5, input_dim=2, activation='sigmoid', kernel_initializer='he_uniform'))
model.add(Dense(5,activation='sigmoid', kernel_initializer='he_uniform'))'''
#model.add(Dense(5,activation='sigmoid', kernel_initializer='he_uniform'))
#model.add(Dense(5,activation='sigmoid', kernel_initializer='he_uniform'))
#model.add(Dense(5,activation='sigmoid', kernel_initializer='he_uniform'))
#model.add(Dense(1, activation='sigmoid'))

#2.增加更多层
#model.add(Dense(5,activation='sigmoid', kernel_initializer=init))
#model.add(Dropout(0.1))
#model.add(Dense(5,activation='sigmoid', kernel_initializer=init))
#model.add(Dropout(0.1))
#model.add(Dense(5,activation='sigmoid', kernel_initializer=init))
#model.add(Dropout(0.1))
#model.add(Dense(5,activation='sigmoid', kernel_initializer=init))
#model.add(Dropout(0.1))
#model.add(BatchNormalization())
'''model.add(Dense(5,activation='sigmoid', kernel_initializer=init))
#model.add(BatchNormalization())
model.add(Dense(5,activation='sigmoid', kernel_initializer=init))
#model.add(BatchNormalization())'''
#model.add(Dense(5,activation='sigmoid', kernel_initializer=init))
#model.add(BatchNormalization())
#model.add(Dense(5,activation='sigmoid', kernel_initializer=init))
#model.add(Dense(5,activation='sigmoid', kernel_initializer=init))

#model.add(Dense(1, activation='sigmoid', kernel_initializer=init))
# compile model
print(model.summary())
save_model(model,model_path)
trained_model,history,_=model_train(model=model,optimizer=opt,loss=loss,dataset=dataset,iters=epoch,batch_size=batch_size,log_path=log_path,callbacks=callbacks,verb=1)
issue_list=determine_issue(history,trained_model,threshold_low=1e-3,threshold_high=1e+3)
issue_list=['vanish']
rm=Repair_Module(model,config,issue_list)
result=rm.solve()
#result_dic=read_csv(log_path,epoch)
#generate_fig(result_dic,fig_name)