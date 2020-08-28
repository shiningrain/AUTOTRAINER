import sys
sys.path.append('../../data')
sys.path.append('../../utils')
from simplednn import load_data
from utils import *
import numpy as np
import keras
from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,Input
from keras.models import Sequential
from keras.optimizers import SGD
from keras.initializers import RandomUniform
from keras.layers.normalization import BatchNormalization
from matplotlib import pyplot
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.layers import Activation
import pandas
import keras
n_hwidth = 5
n_layer = 3
activation = 'relu'
def create_layer(input,activation=activation,layers=n_layer,hwidth=n_hwidth): 
    x=input
    for i in range(layers):
        x = Dense(hwidth,activation=activation)(x)
        #x = BatchNormalization()(x)
        print('Layer %d added' % (i+1))
    return x
def create_layer_skip(input,activation=activation,layers=n_layer,hwidth=n_hwidth):#redusial network
    x=input
    temp_x = x
    for i in range(layers):
        if(i%2!=0):
            temp_x = x
        x = Dense(hwidth)(x)
        #x = BatchNormalization()(x)
        if(i %2 ==0 and i !=0 ):
            x = keras.layers.add([temp_x, x])
            print('Residual layer %d ADD'%i)
        x = Activation(activation)(x)
        print('Layer %d added' % (i))

    return x

# generate 2d classification dataset
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)
# split into train and test
n_train = 800
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
inputs = Input(shape=(2,))
opt = SGD(lr=0.01, momentum=0.9)
x_all=create_layer(input=inputs,activation=activation)
predictions = Dense(1, activation='sigmoid')(x_all)
model = Model(inputs=inputs, outputs=predictions)
model.summary()

loss='binary_crossentropy'
dataset={}
dataset['x']=trainX
dataset['y']=trainy
dataset['x_val']=testX
dataset['y_val']=testy
epoch=300
batch_size=128
log_dir='../../log/case14_circle/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path=log_dir+'case14_circle.csv'
fig_name=log_dir+'case14_circle.pdf'
callbacks=[]

save_model(model,'/data/zxy/DL_tools/DL_tools/models/DyingReLU/case14_circle.h5')
gradient_list1,layer_outputs1,wts1=gradient_test(model,dataset,batch_size)
model,history=model_train(model=model,optimizer=opt,loss=loss,dataset=dataset,iters=epoch,batch_size=batch_size,log_path=log_path,callbacks=callbacks,verb=1)
gradient_list,layer_outputs,wts=gradient_test(model,dataset,batch_size)
tmp_list=[]
for i in range(len(gradient_list)):
    zeros=np.sum(gradient_list[i]==0)
    tmp_list.append(zeros/gradient_list[i].size)
ave_g,_=average_gradient(gradient_list)
issue_list=determine_issue(gradient_list,history,layer_outputs,model,threshold_low=1e-3,threshold_high=1e+3)
result_dic=read_csv(log_path,epoch)
generate_fig(result_dic,fig_name)
print('finish')
# compile model
'''
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit model

csv_name = './activation=%s_layers=%s_hwidth=%s.csv' %(  activation,n_layer,n_hwidth)
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=1000, verbose=1)
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot training history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
pandas.DataFrame(history.history).to_csv(csv_name)
'''