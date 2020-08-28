from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,Input
from keras.models import Sequential
from keras.optimizers import Adam,SGD
from keras.initializers import RandomUniform
from keras.layers.normalization import BatchNormalization
from matplotlib import pyplot
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.layers import Activation
from keras.callbacks import LambdaCallback
import numpy as np
import pandas
import keras
import keras.backend as K
import tensorflow as tf
import os,psutil
import time
from keras.layers import LeakyReLU
from keras.layers.core import Lambda
starttime = time.time()
n_hwidth = 5
n_layer = 5
activation = 'exponential'
#call back check gradient 
class LossHistory(keras.callbacks.Callback):
    def __init__(self,training_data): 
        trainX = training_data[0]
        trainy = training_data[1]
    def on_train_begin(self,logs=None):
        outputTensor = model.output# Or model.layers[index].output
        listOfVariableTensors = model.trainable_weights
        #or variableTensors = model.trainable_weights[0]
        self.gradients = K.gradients(outputTensor,listOfVariableTensors)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    def on_batch_begin(self,batch,logs={}):
        trainingExample = trainX[batch*32]
        evaluated_gradients = self.sess.run(self.gradients,feed_dict={model.input:[trainingExample]})
        for i in range(len(evaluated_gradients)):
            print(i,batch,evaluated_gradients[i].shape,np.sum(evaluated_gradients[i]==0))
        #print(len(evaluated_gradients))
        #print(evaluated_gradients)
        #print(len())
        #print(evaluated_gradients)
    #def on_batch_end(self,batch,logs={}):    
        #outputTensor = model.output# Or model.layers[index].output
        #listOfVariableTensors = model.trainable_weights
        #print(batch,trainX[batch],trainy[batch])
#DNN_Hidden_layers
def create_layer(input,activation=activation,layers=n_layer,hwidth=n_hwidth): 
    x=input
    for i in range(layers):
        #x = BatchNormalization()(x)
        x = Dense(hwidth,activation=activation)(x)
        #x = LeakyReLU()(x)
        print('Layer %d added' % (i+1))
    return x
#DNN_Hidden_redusial_layers
def create_residual_layer(input,activation=activation,layers=n_layer,hwidth=n_hwidth): 
    x=input
    temp_x = input
    for i in range(layers):
        if(i%2!=0):
            temp_x = Lambda(lambda x:-x)(x)
        x = Dense(hwidth)(x)
        if(i %2 ==0 and i !=0 ):
            x = keras.layers.add([temp_x,x])
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
opt = SGD(clipnorm=1.0)
x_all=create_layer(input=inputs,activation=activation,layers=n_layer,hwidth=n_hwidth)
predictions = Dense(1, activation='sigmoid')(x_all)
model = Model(inputs=inputs, outputs=predictions)
model.summary()
# compile model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit model

#callbacks = LambdaCallback(on_batch_begin=lambda batch,logs: print(K.gradients(model.input,model.output)))
#history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=10, verbose=1,callbacks=[LossHistory(training_data=[trainX, trainy])])
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=1)
process = psutil.Process(os.getpid())
print('Used Memory:',process.memory_info().rss / 1024 / 1024,'MB')
endtime = time.time()
dtime = endtime - starttime
print("code running time：%.8s s" % dtime)