# Gradient vanish-RNN
import sys
sys.path.append('../../data')
sys.path.append('../../utils')
from imdb import load_data,preprocess
from utils import *
import numpy as np
import keras
from keras.layers import Dense,SimpleRNN,Dropout,BatchNormalization,LSTM,Embedding
from keras.models import Sequential
from keras.optimizers import SGD,Adam
from keras.initializers import RandomUniform
from keras import backend as K

stringx='case16_LSTM(8LSTM_sigmoid_batch128_adam)_IMDB'
epoch=20
labels=10

max_features = 20000
# cut texts after this number of words (among top max_features most common words)
maxlen =500

(x, y), (x_val, y_val)=load_data(max_features)
x=preprocess(x,maxlen,'tensorflow')
x_val=preprocess(x_val,maxlen,'tensorflow')
x=x[:5000,:]
x_val=x_val[:5000,:]
y=y[:5000,]
y_val=y_val[:5000,]

opt='Adam'
loss='binary_crossentropy'
dataset={}
dataset['x']=x
dataset['y']=y
dataset['x_val']=x_val
dataset['y_val']=y_val
epoch=20
batch_size=256
log_dir='../../log/case16_LSTM_tmp/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path=log_dir+'case16_LSTM.csv'
fig_name=log_dir+'case16_LSTM.pdf'
callbacks=[]
model_path='/data/zxy/DL_tools/DL_tools/models/GradientVanish/case16_LSTM.h5'

print('Build model...')
#init = RandomUniform(minval=0, maxval=1)
#init='glorot_uniform'
init='he_uniform'
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, activation='sigmoid',kernel_initializer=init, recurrent_initializer='orthogonal', bias_initializer='zeros',dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
#model.add(BatchNormalization())

model.add(LSTM(128, activation='sigmoid',kernel_initializer=init, recurrent_initializer='orthogonal', bias_initializer='zeros',dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
#model.add(BatchNormalization())

model.add(LSTM(128, activation='sigmoid',kernel_initializer=init, recurrent_initializer='orthogonal', bias_initializer='zeros',dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
#model.add(BatchNormalization())
model.add(LSTM(128, activation='sigmoid',kernel_initializer=init, recurrent_initializer='orthogonal', bias_initializer='zeros',dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
#model.add(BatchNormalization())
model.add(LSTM(128, activation='sigmoid',kernel_initializer=init, recurrent_initializer='orthogonal', bias_initializer='zeros',dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
#model.add(BatchNormalization())
model.add(LSTM(128, activation='sigmoid',kernel_initializer=init, recurrent_initializer='orthogonal', bias_initializer='zeros',dropout=0.2, recurrent_dropout=0.2,return_sequences=True))

#model.add(BatchNormalization())
model.add(LSTM(128, activation='sigmoid',kernel_initializer=init, recurrent_initializer='orthogonal', bias_initializer='zeros',dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
#model.add(BatchNormalization())
model.add(LSTM(128, activation='sigmoid',kernel_initializer=init, recurrent_initializer='orthogonal', bias_initializer='zeros',dropout=0.2, recurrent_dropout=0.2))
#model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

'''
# define model
#init = RandomUniform(minval=0, maxval=1)
init='glorot_uniform'
#init='he_uniform'
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32,activation='sigmoid',kernel_initializer=init, recurrent_initializer='orthogonal', bias_initializer='zeros',return_sequences=True))
#model.add(BatchNormalization())
model.add(Dropout(0.25))
#model.add(SimpleRNN(32,activation='sigmoid',kernel_initializer=init, recurrent_initializer='orthogonal', bias_initializer='zeros',return_sequences=True))
#model.add(BatchNormalization())
#model.add(Dropout(0.25))
model.add(SimpleRNN(128,activation='sigmoid',kernel_initializer=init, recurrent_initializer='orthogonal', bias_initializer='zeros',return_sequences=True))
#model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(SimpleRNN(128,activation='sigmoid',kernel_initializer=init, recurrent_initializer='orthogonal', bias_initializer='zeros',return_sequences=True))
#model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(SimpleRNN(128,activation='sigmoid',kernel_initializer=init, recurrent_initializer='orthogonal', bias_initializer='zeros'))
#model.add(BatchNormalization())

#model.add(Dropout(0.25))
model.add(Dense(1, activation = 'sigmoid'))
'''
print(model.summary())
#save_model(model,model_path)
model_train(model=model,optimizer=opt,loss=loss,dataset=dataset,iters=epoch,batch_size=batch_size,log_path=log_path,callbacks=callbacks,verb=1)
#result_dic=read_csv(log_path,epoch)
#generate_fig(result_dic,fig_name)
