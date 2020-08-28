from keras.models import load_model
from keras.models import Model
from keras.activations import relu,sigmoid,elu,linear,selu
from keras.layers import BatchNormalization,GaussianNoise,Dropout
from keras.layers import Activation,Add,Dense
from keras.layers.core import Lambda
from keras.initializers import he_uniform,glorot_uniform,zeros
from keras.regularizers import l2,l1,l1_l2
from keras.layers import Input
import keras
def insert_intermediate_layer_in_keras(model, layer_id, new_layer):
    layers = [l for l in model.layers]
    x = layers[0].output
    for i in range(1,len(layers)):
        if i == layer_id:
            x = new_layer(x)
        x = layers[i](x)
    new_model = Model(input=layers[0].input, output=x)
    return new_model
def Gaussion_Noise(model,stddev=0.1):
    model = insert_intermediate_layer_in_keras(model,1,GaussianNoise(stddev))
    return model 
def Modify_Regularizer(model,kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)):
    layers_num = len(model.layers)
    for i in range(layers_num-1):#the last layer don't modify
        if 'kernel_regularizer' in model.layers[i].get_config():
            model.layers[i].kernel_regularizer=kernel_regularizer
        if 'bias_regularizer' in model.layers[i].get_config():
            model.layers[i].kernel_regularizer=bias_regularizer
    return model
def Callback_Early():
    early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=3, verbose=0, mode='auto',baseline=None, restore_best_weights=False)
    return early_stopping
def Dropout_network(model,layer_name = 'dense',rate=0.5):
    layers = [l for l in model.layers]
    x = layers[0].output
    for i in range(1,len(layers)-1):
        x = layers[i](x)
        if layer_name in  model.layers[i].get_config()['name']:
            x = Dropout(rate)(x)
    x = layers[len(layers)-1](x)
    new_model = Model(input=layers[0].input, output=x)
    return new_model
model_name = "case_22.h5" 
model = load_model(model_name)
model = Gaussion_Noise(model)
model.summary()
model.save("r_case_22.h5")
'''
model = load_model('r_case_22.h5')
model.summary()
for layer in model.layers:
    print(layer.get_config())
'''