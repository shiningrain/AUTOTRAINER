from keras.models import load_model
from keras.models import Model
from keras.activations import relu,sigmoid,elu,linear,selu
from keras.layers import BatchNormalization,GaussianNoise
from keras.layers import Activation,Add,Dense
from keras.layers.core import Lambda
from keras.initializers import he_uniform,glorot_uniform,zeros
'''
    DNN_skip_connect,redusial_network_resolution,Generally performed on the dense layer
    
    BN_network,redusial_network_resolution,Add BatchNormalization layer before dense layer
    
    modify_initializer,kernel_initializer is generally used he_uniform,bias_initializer 
    is generally used zeros()

    modify_activations(model,activation),modify layers activations but last layer

    Gaussion_Noise,add GaussianNoise layer after first layer
'''
def insert_intermediate_layer_in_keras(model, layer_id, new_layer):
    layers = [l for l in model.layers]
    x = layers[0].output
    for i in range(1,len(layers)):
        if i == layer_id:
            x = new_layer(x)
        x = layers[i](x)
    new_model = Model(input=layers[0].input, output=x)
    return new_model
def DNN_skip_connect_pre(model,layer_name='dense'):
    layers = [l for l in model.layers]
    x = layers[0].output
    for i in range(1,len(layers)):
        if layer_name in  model.layers[i].get_config()['name'] and 'activation' in model.layers[i].get_config() and layers[i].get_config()['activation']=='relu':
            layers[i].activation=linear
            x = layers[i](x)
            x = Activation('relu')(x)
        else:
            x = layers[i](x)
    new_model = Model(input=layers[0].input, output=x)
    return new_model
def modify_initializer(model):
    layers_num = len(model.layers)
    bias_initializer = zeros()
    kernel_initializer = he_uniform()
    for i in range(layers_num):#the last layer don't modify
        if 'kernel_initializer' in model.layers[i].get_config():
            model.layers[i].kernel_initializer=kernel_initializer
        if 'bias_initializer' in model.layers[i].get_config():
            model.layers[i].bias_initializer=bias_init
    return model
def modify_activations(model,activation):#https://github.com/keras-team/keras/issues/9370
    layers_num = len(model.layers)
    for i in range(layers_num-1):#the last layer don't modify
        if 'activation' in model.layers[i].get_config() and model.layers[i].get_config()['activation']!='linear':
            model.layers[i].activation=activation
    return model 
def BN_network(model):
    layers_num = len(model.layers)
    i=0
    while(i<layers_num-1):#the last layer don't add BN layer
        if 'dense' in model.layers[i].get_config()['name']:
            model = insert_intermediate_layer_in_keras(model,i,BatchNormalization())
            i+=1
            layers_num+=1
        i+=1
    return model
def Gaussion_Noise(model,stddev=0.1):
    model = insert_intermediate_layer_in_keras(model,1,GaussianNoise(stddev))
    return model 
def DNN_skip_connect(model,layer_name='dense'):#only activation
    model = DNN_skip_connect_pre(model)
    layers = [l for l in model.layers]
    x = layers[0].output
    temp_x = layers[0].output
    j=0#dense number
    for i in range(1,len(layers)):
        if layer_name in layers[i].get_config()['name']:
            print(j)
            if j%2 != 0:
                temp_x = x
            j+=1
        if j%2 != 0 and j!=1 and 'activation' in layers[i].get_config()['name'] and layers[i].get_config()['activation']=='relu':
            x = Add()([temp_x,x])
        x = layers[i](x)
    new_model = Model(input=layers[0].input, output=x)
    return new_model 
model_name = "circle_case.h5" 
model = load_model(model_name)

model = Gaussion_Noise(model)
model.summary()
#for layer in model.layers:
    #print(layer.get_config())
#model.save('repair_circle_case.h5')