# mlp with weight regularization for the moons dataset
from sklearn.datasets import make_moons
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2
import keras
from keras.layers import GaussianNoise,Dropout
from keras.models import Model
from keras.models import load_model
# generate 2d classification dataset
early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                              patience=3, verbose=0, mode='auto',
                              baseline=None, restore_best_weights=False)
X, y = make_moons(n_samples=100, noise=0.2, random_state=1)
# split into train and test
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
model = load_model('r_case_22.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainy, epochs=4000, verbose=2,validation_data=(testX, testy))
#model.fit(trainX, trainy, epochs=4000, verbose=2,validation_data=(testX, testy),callbacks=[early_stopping])
# evaluate the model
#_, train_acc = model.evaluate(trainX, trainy, verbose=0)
#_, test_acc = model.evaluate(testX, testy, verbose=0)
#print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))