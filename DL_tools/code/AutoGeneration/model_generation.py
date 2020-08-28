'''from keras.models import load_model
from keras import backend as K
from keras.engine import InputLayer
import coremltools

model = load_model('your_model.h5')

# Create a new input layer to replace the (None,None,None,3) input layer :
input_layer = InputLayer(input_shape=(272, 480, 3), name="input_1")

# Save and convert :
model.layers[0] = input_layer
model.save("reshaped_model.h5")    
coreml_model = coremltools.converters.keras.convert('reshaped_model.h5')    
coreml_model.save('MyPredictor.mlmodel')'''
'''
from tensorflow.keras.datasets import mnist

import autokeras as ak

# Prepare the dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)
print(y_train[:3])  # array([7, 2, 1], dtype=uint8)

# Initialize the ImageClassifier.
clf = ak.ImageClassifier(max_trials=3)
# Search for the best model.
clf.fit(x_train, y_train, epochs=10)
# Evaluate on the testing data.
print('Accuracy: {accuracy}'.format(
    accuracy=clf.evaluate(x_test, y_test)))

# unsupervised greedy layer-wise pretraining for blobs classification problem
from sklearn.datasets import make_blobs
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from matplotlib import pyplot

# prepare the dataset
def prepare_data():
	# generate 2d classification dataset
	X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
	# one hot encode output variable
	y = to_categorical(y)
	# split into train and test
	n_train = 500
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, testX, trainy, testy

# define, fit and evaluate the base autoencoder
def base_autoencoder(trainX, testX):
	# define model
	model = Sequential()
	model.add(Dense(10, input_dim=2, activation='sigmoid', kernel_initializer='he_uniform'))
	model.add(Dense(2, activation='linear'))
	# compile model
	model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9))
	# fit model
	model.fit(trainX, trainX, epochs=100, verbose=0)
	# evaluate reconstruction loss
	train_mse = model.evaluate(trainX, trainX, verbose=0)
	test_mse = model.evaluate(testX, testX, verbose=0)
	print('> reconstruction error train=%.3f, test=%.3f' % (train_mse, test_mse))
	return model

# evaluate the autoencoder as a classifier
def evaluate_autoencoder_as_classifier(model, trainX, trainy, testX, testy):
	# remember the current output layer
	output_layer = model.layers[-1]
	# remove the output layer
	model.pop()
	# mark all remaining layers as non-trainable
	for layer in model.layers:
		layer.trainable = False
	# add new output layer
	model.add(Dense(3, activation='softmax'))
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])
	# fit model
	model.fit(trainX, trainy, epochs=100, verbose=0)
	# evaluate model
	_, train_acc = model.evaluate(trainX, trainy, verbose=0)
	_, test_acc = model.evaluate(testX, testy, verbose=0)
	# put the model back together
	model.pop()
	model.add(output_layer)
	model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9))
	return train_acc, test_acc

# add one new layer and re-train only the new layer
def add_layer_to_autoencoder(model, trainX, testX):
	# remember the current output layer
	output_layer = model.layers[-1]
	# remove the output layer
	model.pop()
	# mark all remaining layers as non-trainable
	for layer in model.layers:
		layer.trainable = False
	# add a new hidden layer
	model.add(Dense(10, activation='sigmoid', kernel_initializer='he_uniform'))
	# re-add the output layer
	model.add(output_layer)
	# fit model
	model.fit(trainX, trainX, epochs=100, verbose=0)
	# evaluate reconstruction loss
	train_mse = model.evaluate(trainX, trainX, verbose=0)
	test_mse = model.evaluate(testX, testX, verbose=0)
	print('> reconstruction error train=%.3f, test=%.3f' % (train_mse, test_mse))

# prepare data
trainX, testX, trainy, testy = prepare_data()
# get the base autoencoder
model = base_autoencoder(trainX, testX)
# evaluate the base model
scores = dict()
train_acc, test_acc = evaluate_autoencoder_as_classifier(model, trainX, trainy, testX, testy)
print('> classifier accuracy layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
scores[len(model.layers)] = (train_acc, test_acc)
# add layers and evaluate the updated model
n_layers = 5
for _ in range(n_layers):
	# add layer
	add_layer_to_autoencoder(model, trainX, testX)
	# evaluate model
	train_acc, test_acc = evaluate_autoencoder_as_classifier(model, trainX, trainy, testX, testy)
	print('> classifier accuracy layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
	# store scores for plotting
	scores[len(model.layers)] = (train_acc, test_acc)
# plot number of added layers vs accuracy
keys = list(scores.keys())
pyplot.plot(keys, [scores[k][0] for k in keys], label='train', marker='.')
pyplot.plot(keys, [scores[k][1] for k in keys], label='test', marker='.')
pyplot.legend()
pyplot.show()'''

import sys
sys.path.append('.')
sys.path.append('../../data')
sys.path.append('../../utils')
sys.path.append('../../configure')
from utils import *
import numpy as np
import copy
from params_config import *
from configure import *
import argparse
import keras
from keras.models import load_model,clone_model
from keras.layers import Dense
from keras.models import Sequential
from keras.models import Model

strategy=['Add_Layers','Modify_Activation','Modify_Initializer']

def get_strategy():
	"""
	return a list of the chosen strategies.
	"""
	tmp_amount=np.random.randint(1,(len(strategy)+1))
	chosen=np.random.choice(strategy,tmp_amount,replace=False)
	return chosen

def replace_intermediate_layer_in_keras(model, layer_id, new_layer):
	layers = [l for l in model.layers]
	x = layers[0].output
	for i in range(1,len(layers)):
		if i == layer_id:
			x = new_layer(x)
		else:
			x = layers[i](x)
	new_model = Model(input=layers[0].input, output=x)
	return new_model

def insert_intermediate_layer_in_keras(model, layer_id, new_layer):
	layers = [l for l in model.layers]
	x = layers[0].output
	for i in range(1,len(layers)):
		if i == layer_id:
			x = new_layer(x)
		x = layers[i](x)
	new_model = Model(input=layers[0].input, output=x)
	return new_model



def random_add_layers(source_model):
	available_layer_type=['simplernn','lstm','dense']#'conv2d','depthwise_conv2d',当前插入conv2d会影响后面模型的shape，有待解决,'lstm'需要更多更改'dense'
	layers_amount=int(np.random.randint(1,5)*3)# the amount of the added layers
	tmp_layers_list=[]
	for i in range(len(source_model.layers)):
		layer_name=source_model.layers[i].get_config()['name']
		tmp_delete='_'+layer_name.split('_')[-1]
		layer_type=layer_name.replace(tmp_delete,'')
		if layer_type in available_layer_type:
			tmp_layers_list.append(i)
	tmp_layers_list.sort()
	del tmp_layers_list[-1]
	accept=0
	while(accept==0):
		layer_number=int(np.random.choice(tmp_layers_list,1)[0])
		new_model=clone_model(source_model)
		kwargs=source_model.layers[layer_number].get_config()
		accept=1
		if 'return_sequences' in kwargs.keys():
			if kwargs['return_sequences']==False:
				accept=0
	add=1   
	for j in range(layers_amount):
		tmp_name=source_model.layers[layer_number].get_config()['name']
		tmp_delete='_'+tmp_name.split('_')[-1]
		tmp_type=tmp_name.replace(tmp_delete,'')
		kwargs['name']='new'+tmp_name+'_'+str(j)
		if 'lstm' ==tmp_type:
			copy_kwargs=copy.deepcopy(kwargs)
			'''if kwargs['return_sequences']==False:#问题在这里		
				print(j)
				if j==0:
					copy_kwargs['return_sequences']=True
					new_model=replace_intermediate_layer_in_keras(source_model,layer_number,source_model.layers[layer_number].__class__(**copy_kwargs))
				elif j==1:
					copy_kwargs['return_sequences']=True
					new_model=insert_intermediate_layer_in_keras(new_model, layer_number+1+j, source_model.layers[layer_number].__class__(**copy_kwargs))
				else:
					new_model=insert_intermediate_layer_in_keras(new_model, layer_number+1+j, source_model.layers[layer_number].__class__(**copy_kwargs))
			else:'''
			new_model=insert_intermediate_layer_in_keras(new_model, layer_number+1+j, source_model.layers[layer_number].__class__(**copy_kwargs))
		else:
			new_model=insert_intermediate_layer_in_keras(new_model, layer_number+1+j, source_model.layers[layer_number].__class__(**kwargs))
	new_model.summary()
	return new_model

def random_modify_layers(source_model,method='acti'):
	new_acti=r_activation()
	new_init1=r_initializer()
	new_init2=r_initializer()
	for i in range(len(source_model.layers)):
		config=source_model.layers[i].get_config()
		if method=='acti':
			if 'activation' in config:
				new_config=copy.deepcopy(config)
				new_config['activation']=new_acti
				new_layer=source_model.layers[i].__class__(**new_config)
				new_model=replace_intermediate_layer_in_keras(source_model,i,new_layer)
		elif method=='init':
			if ('bias_initializer' in config) or ('kernel_initializer' in config):
				new_config=copy.deepcopy(config)
				new_config['bias_initializer']=new_init1
				new_config['kernel_initializer']=new_init2
				new_layer=source_model.layers[i].__class__(**new_config)
				new_model=replace_intermediate_layer_in_keras(source_model,i,new_layer)
	return new_model


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Model Generation')
	parser.add_argument('-model','-m',default='/data/zxy/DL_tools/Training_Automatic_Repaire_Tool/DL_tools/models/seed_model/lstm.h5', help='model path')
	parser.add_argument('--amount', '-am', type=int, default=30, help='The number of generated models')
	parser.add_argument('--save_dir', '-f', type=str, default='/data/zxy/DL_tools/DL_tools/code/AutoGeneration/lstm/model', help='The path to save model')
	args = parser.parse_args()

	#model_seed=load_model(args.model)
	#model_seed.summary()

	for iters in range(args.amount):
		keras.backend.clear_session()
		now_strat=get_strategy()
		#now_strat=strategy
		#model_seed=clone_model(model_seed)
		model_new=load_model(args.model)
		add_message='_'
		print('--------------------------',iters+1)
		if 'Modify_Activation' in now_strat:
			model_new=random_modify_layers(model_new,'acti')
			add_message=add_message+'ma_'
		if 'Modify_Initializer' in now_strat:
			model_new=random_modify_layers(model_new,'init')
			add_message=add_message+'mi_'
		if 'Add_Layers' in now_strat:
			model_new=random_add_layers(model_new)
			add_message=add_message+'al_'
		save_path(args.save_dir,add_message=add_message,model=model_new,method='model')
		
	print('finish')
