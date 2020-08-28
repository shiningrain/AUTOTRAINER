import sys
sys.path.append('../../data')
sys.path.append('../../utils')
sys.path.append('../../modules')
from simplednn import load_data
from utils import *
from modules import *
import numpy as np
import keras
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,Input,LeakyReLU
from sklearn.model_selection import train_test_split
import datetime


#i='RandomNormal'
#i= 'he_uniform'
(x,y),( x_val, y_val )= load_data('circle')

opt='Adam'
loss='binary_crossentropy'
dataset={}
dataset['x']=x
dataset['y']=y
dataset['x_val']=x_val
dataset['y_val']=y_val
epoch=50
batch_size=128
log_dir='../../log/case14/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path=log_dir+'case14_tmp.csv'
fig_name=log_dir+'case14_tmp.pdf'
callbacks=[]
model_path='/data/zxy/DL_tools/DL_tools/models/DyingReLU/case14_tmp.h5'

config={}
config['opt']=opt
config['loss']=loss
config['dataset']=dataset
config['epoch']=epoch
config['batch_size']=batch_size
config['callbacks']=callbacks

init ='RandomUniform'#keras.initializers.RandomUniform(minval=-1,maxval=1,seed=None)#'he_uniform'#'lecun_normal'#'RandomUniform'#
init1 ='Zeros'

inp=Input(shape=(2,))
x=Dense(5,activation='relu', kernel_initializer=init,bias_initializer=init1)(inp)
x=Dense(5, activation='relu', kernel_initializer=init,bias_initializer=init1)(x)
x=Dense(5, activation='relu', kernel_initializer=init,bias_initializer=init1)(x)
x=Dense(5, activation='relu', kernel_initializer=init,bias_initializer=init1)(x)
x=Dense(5,activation='relu', kernel_initializer=init,bias_initializer=init1)(x)
x=Dense(5, activation='relu', kernel_initializer=init,bias_initializer=init1)(x)
x=Dense(5, activation='relu', kernel_initializer=init,bias_initializer=init1)(x)
x=Dense(5, activation='relu', kernel_initializer=init,bias_initializer=init1)(x)

x=Dense(1,activation='sigmoid')(x)
model=Model(inp, x)

'''dense_1=keras.layers.Dense(64,activation='selu',kernel_initializer='lecun_normal',bias_initializer='lecun_normal')(inp)
alp_drop=keras.layers.AlphaDropout(0.25)(dense_1)

dense_1=keras.layers.Dense(64,activation = 'linear',kernel_initializer='lecun_normal',bias_initializer='lecun_normal')(inp)

dense_2=keras.layers.Dense(64,activation = 'linear',kernel_initializer='lecun_normal',bias_initializer='lecun_normal')(dense_1)
dense_3=keras.layers.Dense(64,activation = 'linear',kernel_initializer='lecun_normal',bias_initializer='lecun_normal')(dense_2)
#dense_2=keras.layers.selu()(dense_1)
#dense_3=keras.layers.Activation('selu')(dense_2)

#dense_1 = keras.layers.Dense(64, activation='selu', kernel_initializer=init, bias_initializer=init)(inp)
#outputtensor=dense_1.output
#leaky_selu=keras.layers.advanced_activations.Leakyselu(alpha=0.3)(dense_1)
'''
'''
model=Sequential()
model.add(InputLayer(input_shape = (2,)))
model.add(Dense(64, activation = 'linear',kernel_initializer=init,bias_initializer=init1))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(64, activation = 'linear',kernel_initializer=init,bias_initializer=init1))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(64, activation = 'linear',kernel_initializer=init,bias_initializer=init1))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(64, activation = 'linear',kernel_initializer=init,bias_initializer=init1))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(64, activation = 'linear',kernel_initializer=init,bias_initializer=init1))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(64, activation = 'linear',kernel_initializer=init,bias_initializer=init1))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(64, activation = 'linear',kernel_initializer=init,bias_initializer=init1))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(64, activation = 'linear',kernel_initializer=init,bias_initializer=init1))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(64, activation = 'linear',kernel_initializer=init,bias_initializer=init1))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(64, activation = 'linear',kernel_initializer=init,bias_initializer=init1))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(64, activation = 'linear',kernel_initializer=init,bias_initializer=init1))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(1, activation = 'sigmoid'))
'''

print(model.summary())
save_model(model,model_path)
trained_model,history,_=model_train(model=model,optimizer=opt,loss=loss,dataset=dataset,iters=epoch,batch_size=batch_size,log_path=log_path,callbacks=callbacks,verb=1)
issue_list=determine_issue(history,trained_model,threshold_low=1e-3,threshold_high=1e+3)
issue_list=['relu']
rm=Repair_Module(model,config,issue_list)
result=rm.solve()
result_dic=read_csv(log_path,epoch)
generate_fig(result_dic,fig_name)
print('finish')
