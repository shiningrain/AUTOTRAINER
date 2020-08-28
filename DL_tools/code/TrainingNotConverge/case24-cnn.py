# study of learning rate on accuracy for blobs problem
from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD,Adam
from keras.initializers import RandomUniform
from matplotlib import pyplot
import keras
import datetime
import csv
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from keras import backend as K
from keras.datasets import cifar10
from TimeCounter import TimeHistory
import pandas as pd

epoch=200
# prepare train and test dataset
stringx='case24-cnn-adam'
def readcsv(num,lrate,type0):
    list1=["./log_case24-cnn-adam1.0.csv","./log_case24-cnn-adam0.1.csv","./log_case24-cnn-adam0.01.csv","./log_case24-cnn-adam0.001.csv","./log_case24-cnn-adam0.0001.csv","./log_case24-cnn-adam1e-05.csv","./log_case24-cnn-adam1e-06.csv","./log_case24-cnn-adam1e-07.csv"]
    csvFile = open(list1[num], 'r')
    reader = csv.reader(csvFile)
    result = []
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        item = [float(x) for x in item]
        result.append(item)
    csvFile.close()
    x_axis = []
    for i in range(epoch):
        x_axis.append(i)
    while ([] in result):
            result.remove([])
    result = np.array(result)
    #dtloss=pd.DataFrame({'x': x_axis, 'y': result[:, 2]})
    #dtvloss=pd.DataFrame({'x': x_axis, 'y': result[:, 0]})
    if type0=="acc":
        dtacc=pd.DataFrame({'x': x_axis, 'y': result[:, 3]})
        dtvacc=pd.DataFrame({'x': x_axis, 'y': result[:, 1]})
    elif type0=="loss":
        dtacc=pd.DataFrame({'x': x_axis, 'y': result[:, 2]})
        dtvacc=pd.DataFrame({'x': x_axis, 'y': result[:, 0]})
    pyplot.plot(dtacc['x'], dtacc['y'], linestyle="-", marker='', markerfacecolor='red', color='red', linewidth=2,
             label='acc')#, data=dtacc
    pyplot.plot(dtvacc['x'], dtvacc['y'] ,linestyle="-", marker='', markerfacecolor='orange', color='orange', linewidth=2,
                label='val_acc')#, data=dtvacc
    pyplot.title('lrate=' + str(lrate), pad=-50)

def prepare_data():
    # generate 2d classification dataset
    mnist = input_data.read_data_sets("/data/zxy/DL_tools/MNIST_data", one_hot=True)
    #1.两环问题的简单模型，精度到0.8左右难以提升
    trainX, trainy = mnist.train.images, mnist.train.labels
    testX, testy = mnist.test.images, mnist.test.labels
    trainX = trainX.reshape(-1, 28, 28, 1).astype('float32')
    testX = testX.reshape(-1, 28, 28, 1).astype('float32')
    trainX=trainX[:1000,:,:,:]
    testX=testX[:300,:,:,:]
    trainy=trainy[:1000,:]
    testy=testy[:300,:]
    return trainX, trainy, testX, testy

# fit a model and plot learning curve
def fit_model(trainX, trainy, testX, testy, lrate):
    # define model
    model = Sequential()
    init='glorot_uniform'
    model.add(Conv2D(filters = 16,
            kernel_size = (3, 3),
            padding = 'same',
            input_shape = (28, 28,1),
            activation = 'relu',kernel_initializer=init))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 36,
                    kernel_size = (3, 3),
                    padding = 'same',
                    activation='relu',kernel_initializer=init))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu',kernel_initializer=init))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation = 'softmax'))

    print(model.summary())
    # compile model
    #opt = SGD(lr=lrate)
    opt = Adam(lr=lrate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=epoch, verbose=2)
    time_callback=TimeHistory()
    time_callback.write_to_csv(history.history,"log_case24-cnn-adam"+str(lrate)+".csv", epoch)
    # plot learning curves
    #pyplot.plot(history.history['accuracy'], label='train',linestyle="-", marker='', markerfacecolor='red', color='red', linewidth=2)
    #pyplot.plot(history.history['val_accuracy'], label='test',linestyle="-", marker='', markerfacecolor='orange', color='orange', linewidth=2)
    #pyplot.plot(history.history['loss'], label='train', linestyle="-", marker='', markerfacecolor='blue',
    #            color='blue', linewidth=2)
    #pyplot.plot(history.history['val_loss'], label='test', linestyle="-", marker='', markerfacecolor='green',
     #           color='green', linewidth=2)
    #pyplot.title('lrate='+str(lrate), pad=-50)

# prepare dataset
trainX, trainy, testX, testy = prepare_data()
# create learning curves for different learning rates
learning_rates = [1E-0, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, 1E-7]
for i in range(len(learning_rates)):
    # determine the plot number
    plot_no = 420 + (i+1)
    pyplot.subplot(plot_no)
    # fit model and plot learning curves for a learning rate
    #fit_model(trainX, trainy, testX, testy, learning_rates[i])
    print('finish ',i)
    readcsv(i,learning_rates[i],"loss")
# show learning curves
pyplot.title('loss for different LR')
pdf_name="loss_"+stringx+'.pdf'
pyplot.savefig(pdf_name)
#pyplot.show()