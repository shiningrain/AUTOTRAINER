import sys
sys.path.append('../../data')
sys.path.append('../../utils')
from imdb import load_data,preprocess
from utils import *
import numpy as np
import keras
from keras.layers import Dense,SimpleRNN,Dropout,BatchNormalization,LSTM,Embedding,GaussianNoise
from keras.models import Sequential
from keras.optimizers import SGD,Adam
from keras.initializers import RandomUniform
from keras import regularizers


stringx='case30_lstm_regualr(1rnn_tanh_batch128_adam0.1lr)_IMDB'
epoch=50
labels=10

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_features = 5000
# cut texts after this number of words (among top max_features most common words)
maxlen =500
batch_size = 256

(x, y), (x_val, y_val)=load_data(max_features)
x=preprocess(x,maxlen,'tensorflow')
x_val=preprocess(x_val,maxlen,'tensorflow')
x=x[:10000,:]
x_val=x_val[:5000,:]
y=y[:10000,]
y_val=y_val[:5000,]

opt='Adam'
loss='binary_crossentropy'
dataset={}
dataset['x']=x
dataset['y']=y
dataset['x_val']=x_val
dataset['y_val']=y_val
epoch=30
batch_size=256
log_dir='../../log/case30_lstm_regualr/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path=log_dir+'case30_lstm_regualr.csv'
fig_name=log_dir+'case30_lstm_regualr.pdf'
callbacks=[]
model_path='/data/zxy/DL_tools/DL_tools/models/case30_lstm_regualr.h5'
#Estop=keras.callbacks.EarlyStopping(monitor='val_loss',patience=2, verbose=1)
#callbacks.append(Estop)

config={}
config['opt']=opt
config['loss']=loss
config['dataset']=dataset
config['epoch']=epoch
config['batch_size']=batch_size
config['callbacks']=callbacks

print('Build model...')
#init = RandomUniform(minval=0, maxval=1)
init='glorot_uniform'

model = Sequential()
model.add(Embedding(max_features, 64))
#model.add(GaussianNoise(stddev=0.1))
model.add(LSTM(64, activation='tanh', recurrent_initializer='orthogonal', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.001)))#,dropout=0.2, recurrent_dropout=0.2))#,dropout=0.2, recurrent_dropout=0.2)
#model.add(SimpleRNN(64, activation='tanh', recurrent_initializer='orthogonal', bias_initializer='zeros'))#,dropout=0.2, recurrent_dropout=0.2))#.,2
model.add(BatchNormalization())
#model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())
#save_model(model,model_path)
trained_model,history,_=model_train(model=model,optimizer=opt,loss=loss,dataset=dataset,iters=epoch,batch_size=batch_size,log_path=log_path,callbacks=callbacks,verb=1)
#issue_list=determine_issue(history,trained_model,threshold_low=1e-3,threshold_high=1e+3)
#issue_list=['vanish']
#rm=Repair_Module(model,config,issue_list)
#result=rm.solve()
result_dic=read_csv(log_path,epoch)
generate_fig(result_dic,fig_name)
print(1)