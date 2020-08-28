import keras
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.callbacks import TensorBoard
from keras.layers import LeakyReLU
from keras.optimizers import SGD,Adam
import pandas
from keras.models import Model
import matplotlib.pyplot as pyplot
# import noise layer
from keras.layers import GaussianNoise
import os,psutil
import time
from keras.layers.core import Lambda
starttime = time.time()
# define noise layer
# Specify number of layers
n_layers = 35

# Width of hidden layer, number of neurons in the hidden layer. all layers have same width. 
n_hwidth = 128
n_classes = 10
epochs = 50
activation='relu'#'selu'
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

inputs = Input(shape=(784,))

def create_Layer(inputs,n_layers,n_hwidth,activation):
    x = (inputs)
    for k in range(n_layers):
        #x = BatchNormalization()(x)
        x = Dense(n_hwidth)(x)
        x = Activation(activation)(x)
        print('Layer %d added' % (k+1))
    return x
def create_residual_layer(input,n_layers,hwidth,activation): 
    x=input
    temp_x = input
    for i in range(n_layers):
        if(i%2!=0):
            temp_x = Lambda(lambda x:x)(x)
        x = Dense(hwidth)(x)
        if(i %2 ==0 and i !=0 ):
            x = keras.layers.add([temp_x,x])
            print('Residual layer %d ADD'%i)
        x = Activation(activation)(x)
        print('Layer %d added' % (i))
    return x
# Create all layers    
x_all = create_Layer(inputs,n_layers,n_hwidth,activation)
# Output layer
predictions = Dense(n_classes, activation='softmax')(x_all)
print('Output layer added')
# Create Model
model = Model(inputs=inputs, outputs=predictions)
# Print model summary
model.summary()
opt = Adam()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

process = psutil.Process(os.getpid())
print('Used Memory:',process.memory_info().rss / 1024 / 1024,'MB')
endtime = time.time()
dtime = endtime - starttime
print("程序运行时间为：%.8s s" % dtime)
'''
ttb_dir = './MLP_SGD_%s_%s' %(n_layers,activation,)
callbacks = [TensorBoard(log_dir=ttb_dir, histogram_freq=0, batch_size=32, write_grads=False)]

history = model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=callbacks)


csv_name = 'MLP_SGD_%s_%s.csv' %(n_layers,activation)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
#pyplot.savefig('MLP_Redsuial_%s_%s.jpeg')
pandas.DataFrame(history.history).to_csv(csv_name)
'''