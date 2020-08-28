# study of learning rate on accuracy for blobs problem
from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization,Embedding,SimpleRNN
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
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

epoch=15
max_features = 5000# 20000 before
# cut texts after this number of words (among top max_features most common words)
maxlen = 500# 80 before
batch_size = 32
# prepare train and test dataset
stringx='case24-rnn-simplernn-sgd'
def readcsv(num,lrate,type0):
    list1=["./log_case24-rnn_sgd_1.0-1.csv","./log_case24-rnn_sgd_0.1-1.csv","./log_case24-rnn_sgd_0.01-1.csv","./log_case24-rnn_sgd_0.001-1.csv","./log_case24-rnn_sgd_0.0001-1.csv","./log_case24-rnn_sgd_1e-05-1.csv","./log_case24-rnn_sgd_1e-06-1.csv","./log_case24-rnn_sgd_1e-07-1.csv"]
    #list1=["./log_case24-rnn_adam_1.0-1.csv","./log_case24-rnn_adam_0.1-1.csv","./log_case24-rnn_adam_0.01-1.csv","./log_case24-rnn_adam_0.001-1.csv","./log_case24-rnn_adam_0.0001-1.csv","./log_case24-rnn_adam_1e-05-1.csv","./log_case24-rnn_adam_1e-06-1.csv","./log_case24-rnn_adam_1e-07-1.csv"]
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
    print('Loading data...')
    (x_train, trainy), (x_test, testy) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    trainX = sequence.pad_sequences(x_train, maxlen=maxlen)
    testX = sequence.pad_sequences(x_test, maxlen=maxlen)
    #trainX=trainX[:5000,:]
    #testX=testX[:5000,:]
    #trainy=trainy[:5000,]
    #testy=testy[:5000,]
    print('x_train shape:', trainX.shape)
    print('x_test shape:', testX.shape)
    return trainX, trainy, testX, testy

# fit a model and plot learning curve
def fit_model(trainX, trainy, testX, testy, lrate):
    # define model
    #init='he_uniform'
    model = Sequential()
    model.add(Embedding(max_features, 64))
    #model.add(LSTM(64, activation='tanh', recurrent_initializer='orthogonal', bias_initializer='zeros',dropout=0.2, recurrent_dropout=0.2))
    model.add(SimpleRNN(64, activation='tanh', recurrent_initializer='orthogonal', bias_initializer='zeros',dropout=0.2, recurrent_dropout=0.2))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())
    # compile model
    #opt = SGD(lr=lrate)
    opt = Adam(lr=lrate)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=epoch, verbose=1)
    time_callback=TimeHistory()
    time_callback.write_to_csv(history.history,"log_case24-rnn_adam_"+str(lrate)+"-1.csv", epoch)
    # plot learning curves
    #pyplot.plot(history.history['accuracy'], label='train',linestyle="-", marker='', markerfacecolor='red', color='red', linewidth=2)
    #pyplot.plot(history.history['val_accuracy'], label='test',linestyle="-", marker='', markerfacecolor='orange', color='orange', linewidth=2)
    #pyplot.plot(history.history['loss'], label='train', linestyle="-", marker='', markerfacecolor='blue',
    #            color='blue', linewidth=2)
    #pyplot.plot(history.history['val_loss'], label='test', linestyle="-", marker='', markerfacecolor='green',
     #           color='green', linewidth=2)
    #pyplot.title('lrate='+str(lrate), pad=-50)

# prepare dataset
#trainX, trainy, testX, testy = prepare_data()
# create learning curves for different learning rates
learning_rates =[1E-0, 1E-1, 1E-2, 1E-3,1E-4, 1E-5, 1E-6, 1E-7]#,#[0.01]#
for i in range(len(learning_rates)):
    print('----------------start------------',learning_rates[i])
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