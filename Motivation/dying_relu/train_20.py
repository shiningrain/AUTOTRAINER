from keras.datasets import mnist
from keras.models import load_model
from keras import backend as K
import pandas
import keras
import sys
def minist_load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = minist_load_data()
epochs = 50
train_num = 100
opt = 'adam'
for i in range(train_num):
    model = load_model("minist_relu_layer=20.h5")
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=32,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
    csv_name = "case_16_later=20_%s.csv".format(i)
    pandas.DataFrame(history.history).to_csv(csv_name)
    K.clear_session()
