import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
import keras.backend as K
import tensorflow as tf
from keras.callbacks import LambdaCallback
class LossHistory(keras.callbacks.Callback):
    def on_batch_begin(self,epoch,logs={}):    
        outputTensor = model.output# Or model.layers[index].output
        listOfVariableTensors = model.trainable_weights
        #or variableTensors = model.trainable_weights[0]
        gradients = K.gradients(outputTensor,listOfVariableTensors)
        trainingExample = np.random.random((1,1))
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        evaluated_gradients = sess.run(gradients,feed_dict={model.input:trainingExample})
        print(len(evaluated_gradients[4]))
        #print(evaluated_gradients)

n = 100         # sample size
x = np.linspace(0,1,n)    #input
y = 4*(x-0.5)**2          #output
dy = 8*(x-0.5)       #derivative of output wrt the input
model = Sequential()
model.add(Dense(32, input_dim=1, activation='relu'))            # 1d input
model.add(Dense(32, activation='relu'))
model.add(Dense(1))                                             # 1d output
# Minimize mse
model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])
model.summary()
#callbacks = LambdaCallback(on_batch_begin=lambda batch,logs: print(K.gradients(model.input,model.output)))
model.fit(x, y, batch_size=10, epochs=10, verbose=1,callbacks=[LossHistory()])
#model.fit(x, y, batch_size=10, epochs=100, verbose=1)

'''
#value of gradient for the first x_test
sess = tf.Session()
sess.run(tf.global_variables_initializer())
evaluated_gradients_1 = sess.run(gradients[0], feed_dict={model.input: x,model.output:y})
print(evaluated_gradients_1)

#value of gradient for the second x_test
#x_test_2 = np.array([[0.6]])
evaluated_gradients_2 = sess.run(gradients[0], feed_dict={model.input: x[1],model.output:y[1]})
print(evaluated_gradients_2)
'''
'''
foo = [2, 18, 9, 22, 17, 24, 8, 12, 27]
z=filter(lambda x: x % 3 == 0, foo)
x=map(lambda x: x * 2 + 10, foo)
print('finish')
'''