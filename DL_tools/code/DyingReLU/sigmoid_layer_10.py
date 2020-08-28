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
from keras.callbacks import LambdaCallback,hist
import numpy as np
import pandas
import keras
import keras.backend as K
import tensorflow as tf
n_hwidth = 5
n_layer = 5
activation = 'relu'
class LossHistory(keras.callbacks.Callback):
    def __init__(self,training_data): 
        trainX = training_data[0]
        trainy = training_data[1]
    '''def on_train_begin(self,logs=None):
        outputTensor = model.output# Or model.layers[index].output
        listOfVariableTensors = model.trainable_weights
        #or variableTensors = model.trainable_weights[0]
        #self.gradients = K.gradients(outputTensor,listOfVariableTensors)
        #self.sess = tf.InteractiveSession()
        #self.sess.run(tf.global_variables_initializer())
        gradients = K.gradients(outputTensor,listOfVariableTensors)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())'''
    def on_epoch_begin(self,epoch,logs={}):
        outputTensor = model.output# Or model.layers[index].output
        listOfVariableTensors = model.trainable_weights
        #or variableTensors = model.trainable_weights[0]
        #self.gradients = K.gradients(outputTensor,listOfVariableTensors)
        #self.sess = tf.InteractiveSession()
        #self.sess.run(tf.global_variables_initializer())
        gradients = K.gradients(outputTensor,listOfVariableTensors)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        trainingExample = trainX[0]
        evaluated_gradients = sess.run(gradients,feed_dict={model.input:[trainingExample]})
        #for i in range(len(evaluated_gradients)):
        #    print(i,np.max(evaluated_gradients[i]))
        #print(len(evaluated_gradients))
        #print(evaluated_gradients)
        #print(len())
        #print(evaluated_gradients)
        print('-----')
    def on_batch_end(self,batch,logs={}):    
        outputTensor = model.output# Or model.layers[index].output
        listOfVariableTensors = model.trainable_weights
        #print(batch,trainX[batch],trainy[batch])
def create_layer(input,activation=activation,layers=n_layer,hwidth=n_hwidth): 
    x=input
    for i in range(layers):
        x = Dense(hwidth,activation=activation)(x)
        #x = BatchNormalization()(x)
        print('Layer %d added' % (i+1))
    return x
def create_layer_skip(input,activation=activation,layers=n_layer,hwidth=n_hwidth): 
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
# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model

#ttb_dir = './activation=%s_layers=%s_hwidth=%s_skip' %(  activation,n_layer,n_hwidth)'
#csv_name = './activation=%s_layers=%s_hwidth=%s_skip.csv' %(activation,n_layer,n_hwidth)'
#callbacks = LambdaCallback(on_batch_begin=lambda batch,logs: print(K.gradients(model.input,model.output)))
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=1,callbacks=[LossHistory(training_data=[trainX, trainy])])
#history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=1)
'''
#gradients = K.gradients(model.output, model.weights)              #Gradient of output wrt the input of the model (Tensor)
#print(gradients)
outputTensor = model.output# Or model.layers[index].output
listOfVariableTensors = model.trainable_weights
#or variableTensors = model.trainable_weights[0]
gradients = K.gradients(outputTensor,listOfVariableTensors)
trainingExample_x = np.reshape(X,-1,2)
trainingExample_y = np.reshape(y,-1,1)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
evaluated_gradients = sess.run(gradients,feed_dict={model.input:trainingExample_x,model.output:trainingExample_y})
print(evaluated_gradients)
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot training history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
#pandas.DataFrame(history.history).to_csv(csv_name)
'''