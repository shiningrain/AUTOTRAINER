from keras.layers import Dense,Input
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LambdaCallback
from keras.layers import Activation
import keras
import numpy as np
import tensorflow as tf
import pandas as pd 
from keras.models import Model
from keras.optimizers import SGD,RMSprop,Adam
import os,psutil
import time
starttime = time.time()
class LossHistory(keras.callbacks.Callback):
    def __init__(self,training_data): 
        trainX = training_data[0]
        trainy = training_data[1]
    def on_train_begin(self,logs=None):#Training  Initialization
        outputTensor = model.output# Or model.layers[index].output
        listOfVariableTensors = model.trainable_weights
        self.gradients = K.gradients(outputTensor,listOfVariableTensors)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    def on_batch_begin(self,batch,logs={}):#Calculate the gradient of the first element of each batch , Display the maximum gradient value of each layer
        trainingExample = trainX[batch*32]
        evaluated_gradients = self.sess.run(self.gradients,feed_dict={model.input:[trainingExample]})
        for i in range(len(evaluated_gradients)):
            print(i,batch,np.max(evaluated_gradients[i]))
def Hidden_layer(input,activation,layers,hwidth): 
    x=input
    for i in range(layers):
        #x = BatchNormalization()(x)
        x = Dense(hwidth,activation=activation)(x)
        print('Layer %d added' % (i+1))
    return x
def redusial_Hidden_layer(input,activation,layers,hwidth): 
    x=input
    temp_x = x
    for i in range(layers):
        if(i%2!=0):
            temp_x = x
        x = Dense(hwidth,activation=activation)(x)
        if(i %2 ==0 and i !=0 ):
            x = keras.layers.add([temp_x, x])
            print('Residual layer %d ADD'%i)
        x = Activation(activation)(x)
        print('Layer %d added' % (i))
    return x

n_layer = 10
n_hwidth = 31
activation = 'relu'
input_dim = 31
inputs = Input(shape=(input_dim,))
train_len= 3500
test_len = 500
data_num = 4000
epochs=200
data = pd.read_csv('training.csv',nrows=data_num)
data_x=np.zeros((data_num,31))
data_y=np.zeros((data_num,2))
for i in range(data_num):
    data_x[i]=(data.values[i][1:32])
    if data.values[i][32]=='s':
        data_y[i][0]=1
    else: 
        data_y[i][1]=1
x_train=data_x[:train_len]
y_train=data_y[:train_len]
x_test=data_x[train_len:]
y_test=data_y[train_len:]
x_all=Hidden_layer(input=inputs,activation=activation,layers=n_layer,hwidth=n_hwidth)
predictions = Dense(2, activation='softmax')(x_all)
model = Model(inputs=inputs, outputs=predictions)
opt = SGD(0.0001)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
#history = model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=2, validation_data=(x_test, y_test),callbacks=[LossHistory])
history = model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=2, validation_data=(x_test, y_test))
process = psutil.Process(os.getpid())
print('Used Memory:',process.memory_info().rss / 1024 / 1024,'MB')
endtime = time.time()
dtime = endtime - starttime
print("程序运行时间为：%.8s s" % dtime)