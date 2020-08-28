# mlp with unscaled data for the regression problem
from sklearn.datasets import make_regression
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD,adam
from matplotlib import pyplot
import os,psutil
import time
from keras.layers.normalization import BatchNormalization

starttime = time.time()
# generate regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
model = Sequential()
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='linear'))
# compile model
opt = SGD()
model.compile(loss='mean_squared_error', optimizer=opt)
# fit model
model.summary()
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=1,batch_size=32)
# evaluate the model
train_mse = model.evaluate(trainX, trainy, verbose=0)
test_mse = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
process = psutil.Process(os.getpid())
print('Used Memory:',process.memory_info().rss / 1024 / 1024,'MB')
endtime = time.time()
dtime = endtime - starttime
print("程序运行时间为：%.8s s" % dtime)