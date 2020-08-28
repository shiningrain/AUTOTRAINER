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
from keras.models import load_model
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)
# split into train and test
n_train = 800
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
model = load_model('repair_circle_case.h5')
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=1)
