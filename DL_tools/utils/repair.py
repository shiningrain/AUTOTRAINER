import os
import sys
sys.path.append('.')
# from utils import has_NaN
import matplotlib.pyplot as plt
import csv
import numpy as np
import keras
import datetime
from TimeCounter import TimeHistory
from keras.models import load_model,Sequential
import keras.backend as K
import tensorflow as tf
from logger import Logger
import copy
logger = Logger()

from keras.models import load_model
from keras.models import Model
from keras.activations import relu,sigmoid,elu,linear,selu
from keras.regularizers import l2,l1,l1_l2
from keras.layers import BatchNormalization,GaussianNoise,Dropout
from keras.layers import Activation,Add,Dense
from keras.layers.core import Lambda
from keras.initializers import he_uniform,glorot_uniform,zeros
from keras.callbacks.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD, Adam, Adamax
import keras.optimizers as O
import keras.layers as L
import keras.activations as A
import keras.initializers as I

tmp_model_path='/data/zxy/DL_tools/DL_tools/models/tmp_model.h5'

def reload_model(model,path=tmp_model_path):
    model.save(path)
    model=load_model(path)
    return model

def random_kwargs_list(method='gradient_clip',clipnorm=0.3,clipvalue=0.5):
    kwargs_list=[]
    if method=='gradient_clip':
        tmp_list=['clipnorm','clipvalue']
        op_type=np.random.choice(tmp_list,1)[0]
        kwargs_list.append(op_type)
        if op_type=='clipnorm':
            kwargs_list.append(round(np.random.uniform(float(1-clipnorm),float(1+clipnorm)),2))
        if op_type=='clipvalue':
            kwargs_list.append(round(np.random.uniform(float(1.0-clipvalue),1.0),2))
    if method=='momentum':
        tmp_momentum=round(np.random.uniform(0.01,0.9),2)
        kwargs_list.append(tmp_momentum)
    return kwargs_list


def last_layer(layers):
    for i in range(len(layers)):
        tmp_config=layers[len(layers)-1-i].get_config()
        if 'activation' in tmp_config['name']:
            continue
        elif 'activation' in tmp_config:
            return len(layers)-1-i


def insert_intermediate_layer_in_keras(model, layer_id, new_layer):
    layers = [l for l in model.layers]
    x = layers[0].output
    for i in range(1,len(layers)):
        if i == layer_id:
            x = new_layer(x)
        x = layers[i](x)
    new_model = Model(input=layers[0].input, output=x)
    return new_model


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
    '''
def DNN_skip_connect_pre(model,layer_name='dense'):
    layers = [l for l in model.layers]
    x = layers[0].output
    for i in range(1,len(layers)):
        if layer_name in  model.layers[i].get_config()['name'] and 'activation' in model.layers[i].get_config() and layers[i].get_config()['activation']=='relu':
            layers[i].activation=linear
            x = layers[i](x)
            layer_name='new_activation_'+str(i)
            x = Activation('relu',name=layer_name)(x)
        else:
            x = layers[i](x)
    new_model = Model(input=layers[0].input, output=x)
    return new_model
'''

def modify_initializer(seed_model,b_initializer=None,k_initializer=None):
    model=copy.deepcopy(seed_model)
    layers_num = len(model.layers)
    bias_initializer = getattr(I, b_initializer)
    kernel_initializer = getattr(I, k_initializer)
    last=last_layer(model.layers)
    for i in range(int(last)):#the last layer don't modify
        if (kernel_initializer!=None) and ('kernel_initializer' in model.layers[i].get_config()):
            model.layers[i].kernel_initializer=kernel_initializer
            '''new_config=copy.deepcopy(model.layers[i].get_config())
            new_config['kernel_initializer']=k_initializer
            new_layer=model.layers[i].__class__(**new_config)
            model=replace_intermediate_layer_in_keras(model,i,new_layer)'''
        if (bias_initializer!=None) and ('bias_initializer' in model.layers[i].get_config()):
            model.layers[i].bias_initializer=bias_initializer
            '''new_config=copy.deepcopy(model.layers[i].get_config())
            new_config['bias_initializer']=b_initializer
            new_layer=model.layers[i].__class__(**new_config)
            model=replace_intermediate_layer_in_keras(model,i,new_layer)'''
    model=reload_model(model)
    return model

def not_dense_acti(model,i):
    """
    for dense(x)+activation(x)/advanced activation(x), don't modify the activation, just keep dense(linear)+ its activation
    """
    advanced_list=['leaky_re_lu','elu','prelu','softmax','activation','thresholded_re_lu','re_lu']
    for j in range(len(advanced_list)):
        if (i+1)<len(model.layers) and advanced_list[j] in model.layers[i+1].get_config()['name']:
            if model.layers[i].get_config()['activation']!=linear:
                model.layers[i].activation=linear
            return False
        if advanced_list[j] in model.layers[i].get_config()['name']:
            return False
    return True


def modify_activations(model,activation,method='normal'):#https://github.com/keras-team/keras/issues/9370
    """
    normal method: activaiton is a function
    special method activatio is a string
    """
    activation=getattr(A, activation)
    layers_num = len(model.layers)
    if method=='normal':# For relu
        last=last_layer(model.layers)
        for i in range(int(last)):###考虑special情况
            if ('activation' in model.layers[i].get_config()) and not_dense_acti(model,i):
                model.layers[i].activation=activation
    if method=='special':# For leakyrelu and others
        i=0
        #layers_num=int(last_layer(model.layers))
        while(i<layers_num):#the last layer don't add BN layer
            if 'activation' in model.layers[i].get_config():
                if not not_dense_acti(model,i):
                    #print(1)
                    if i+2==layers_num: i+=1# the layer layer activation, then stop while
                else:
                    model.layers[i].activation=linear
                    act_cls = getattr(L, activation)
                    model = insert_intermediate_layer_in_keras(model,i+1,act_cls())
                    i+=1
                    layers_num+=1
            i+=1
        #model.summary()
    model=reload_model(model)
    return model
'''
def modify_activations(seed_model,activation,method='normal'):#https://github.com/keras-team/keras/issues/9370
    model=copy.deepcopy(seed_model)
    layers_num = len(model.layers)
    if method=='normal':# For relu
        last=last_layer(model.layers)
        for i in range(int(last)):
            if 'activation' in model.layers[i].get_config():
                #model.layers[i].activation=activation
                new_config=copy.deepcopy(model.layers[i].get_config())
                new_config['activation']=activation
                new_layer=model.layers[i].__class__(**new_config)
                model=replace_intermediate_layer_in_keras(model,i,new_layer)
                #pass
        wts=model.get_weights()
        for l in range(len(wts)):
            if has_NaN(wts[l]):
                for i in range(int(layers_num-last)):
                    new_config=copy.deepcopy(model.layers[last+i].get_config())
                    new_layer=model.layers[last+i].__class__(**new_config)
                    model=replace_intermediate_layer_in_keras(model,(last+i),new_layer)
                break
        print(1)
    if method=='special':# For leakyrelu and others
        i=0
        #layers_num=int(last_layer(model.layers))
        while(i<layers_num):#the last layer don't add BN layer
            if 'activation' in model.layers[i].get_config():
                if i+1==layers_num:
                    break
                if 'activation' in model.layers[i+1].get_config()['name']:
                    #print(1)
                    if i+2==layers_num: i+=1# the layer layer activation, then stop while
                else:
                    #model.layers[i].activation=linear
                    new_config=copy.deepcopy(model.layers[i].get_config())
                    new_config['activation']='linear'
                    new_layer=model.layers[i].__class__(**new_config)
                    model=replace_intermediate_layer_in_keras(model,i,new_layer)
                    act_cls = getattr(L, activation)
                    model = insert_intermediate_layer_in_keras(model,i+1,act_cls())
                    i+=1
                    layers_num+=1
            i+=1
    model.summary()
    return model 
'''

def modify_regularizer(model,kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)):
    last=last_layer(model.layers)
    for i in range(int(last)):#the last layer don't modify
        if 'kernel_regularizer' in model.layers[i].get_config():
            model.layers[i].kernel_regularizer=kernel_regularizer
        if 'bias_regularizer' in model.layers[i].get_config():
            model.layers[i].kernel_regularizer=bias_regularizer
    model=reload_model(model)
    return model

def Dropout_network(seed_model,incert_layer='dense',rate=0.5):
    model=copy.deepcopy(seed_model)
    layers_num = len(model.layers)
    i=0
    while(i<layers_num-1):#the last layer don't add BN layer
        if incert_layer in model.layers[i].get_config()['name']:
            if model.layers[i+1].__class__!=getattr(L,'Dropout'):
                model = insert_intermediate_layer_in_keras(model,i+1,Dropout(rate=rate))
                i+=1
                layers_num+=1
        i+=1
    model.summary()
    model=reload_model(model)
    return model

def BN_network(seed_model,incert_layer='dense'):
    model=copy.deepcopy(seed_model)
    layers_num = len(model.layers)
    i=0
    while(i<layers_num-1):#the last layer don't add BN layer
        if incert_layer in model.layers[i].get_config()['name']:
            if model.layers[i+1].__class__!=getattr(L,'BatchNormalization'):
                model = insert_intermediate_layer_in_keras(model,i+1,BatchNormalization())
                i+=1
                layers_num+=1
        i+=1
    model.summary()
    model=reload_model(model)
    return model


def Gaussion_Noise(seed_model,stddev=0.1):
    model=copy.deepcopy(seed_model)
    model = insert_intermediate_layer_in_keras(model,1,GaussianNoise(stddev))
    model=reload_model(model)
    return model


def DNN_skip_connect(seed_model,layer_name='dense'):#only activation
    model=copy.deepcopy(seed_model)
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

def modify_optimizer(optimizer,kwargs_list,method='lr'):
    if isinstance(optimizer,str):
        opt_cls = getattr(O, optimizer)
        new_opt = opt_cls()
    if method=='lr':
        current_lr=K.eval(optimizer.lr)
        kwargs=optimizer.get_config()
        kwargs['lr']=kwargs_list[0]*current_lr
        new_opt=optimizer.__class__(**kwargs)
    elif method=='momentum':
        new_opt=SGD(momentum=kwargs_list[0])
        '''elif method=='opt':
            # use a new optimizer, kwargs list contains a optimizer name and a kwargs list now.
            opt_cls = getattr(O, kwargs_list[0])
            new_opt = opt_cls(**kwargs_list[-1])'''
    elif method=='gradient':
        # add gradient clip or gradient norm, kwargs list contains a optimizer name and its kwargs now.
        kwargs=optimizer.get_config()
        kwargs[kwargs_list[0]]=kwargs_list[-1]
        new_opt=optimizer.__class__(**kwargs)
    return new_opt
'''
def modify_batch(batch_size):
    return batch_size'''


'''
model=load_model('/data/zxy/DL_tools/DL_tools/models/seed_model/lenet_seed.h5')
#model_1=modify_initializer(model,b_initializer=he_uniform,k_initializer=he_uniform)
model_1=modify_activations(model,'LeakyReLU',method='special')
model_2=modify_activations(model,'relu')
model_1=DNN_skip_connect(model,layer_name='dense')
#model_1=Gaussion_Noise(model)
#model_1=BN_network(model,'dense')
#model.save('/data/zxy/DL_tools/DL_tools/tmp/model.h5')
model_1=load_model('/data/zxy/DL_tools/DL_tools/tmp/model.h5')
print(1)'''
def repair_strategy(method='balance'):
    if method=='balance':
        #first order: efficiency - complexity:
        #second order: efficiency, last order: complexity.
        gradient_vanish_strategy=['selu_1','relu_1','bn_1']
        gradient_explode_strategy=['leaky_3','relu_1','gradient_2','tanh_1','bn_1']#delete the first one#'selu_1',
        dying_relu_strategy=['selu_1','bn_1','initial_3','leaky_3']#
        unstable_strategy=['adam_1','lr_3','ReduceLR_1','batch_3','momentum_3','GN_1']#
        not_converge_strategy=['optimizer_3','lr_3']
        over_fitting_strategy=['regular_1','estop_1','dropout_1','GN_1']
    elif method=='structure':
        #first order: complexity; seconde order: efficiency.
        gradient_vanish_strategy=['relu_1','selu_1','bn_1']
        gradient_explode_strategy=['gradient_2','relu_1','selu_1','tanh_1','bn_1']#1 ,
        dying_relu_strategy=['selu_1','initial_3','leaky_3','bn_1']#
        unstable_strategy=['adam_1','lr_3','ReduceLR_1','batch_3','momentum_3','GN_1']#
        not_converge_strategy=['optimizer_3','lr_3']
        over_fitting_strategy=['estop_1','regular_1','dropout_1','GN_1']
    elif method=='efficiency':
        #first order: efficiency; seconde order: complexity.
        gradient_vanish_strategy=['relu_1','selu_1','bn_1']
        gradient_explode_strategy=['relu_1','selu_1','tanh_1','gradient_2','bn_1']#1 ,
        dying_relu_strategy=['selu_1','initial_3','leaky_3','bn_1']#
        unstable_strategy=['adam_1','lr_3','ReduceLR_1','batch_3','momentum_3','GN_1']#
        not_converge_strategy=['optimizer_3','lr_3']
        over_fitting_strategy=['regular_1','dropout_1','estop_1','GN_1']
    else:
        print('Not support this method')
        os._exit(0)#you can design your repair strategy here.
    return [gradient_vanish_strategy,gradient_explode_strategy,dying_relu_strategy,unstable_strategy,not_converge_strategy,over_fitting_strategy]

##------------------------add solution describe here-----------------------------
def op_gradient(model, config, issue, j):  #m
    tmp_model = model
    if ('clipvalue'
            in config['opt'].get_config()) or ('clipnorm'
                                               in config['opt'].get_config()):
        return tmp_model, config, True
    kwargs_list = random_kwargs_list(method='gradient_clip')
    config['opt'] = modify_optimizer(config['opt'],
                                     kwargs_list,
                                     method='gradient')
    describe = "Using 'Gradient Clip' operation, add {}={} to the optimizer".format(
        str(kwargs_list[0]), str(kwargs_list[-1]))
    return tmp_model, config,describe, False


def op_relu(model, config, issue, j):  #
    #tmp_model=modify_activations(model,'relu')
    tmp_model = modify_initializer(model, k_initializer='he_uniform')
    tmp_model = modify_activations(model, relu)
    #because of the nan weight, in this step ,the weight of the last layer may still be nan even we changed the activations in the previous layers
    # this may lead to nan loss after repair. **reload the model will avoid this**.
    describe = "Using 'ReLU' activation in each layers' activations; Use 'he_uniform' as the kernel initializer."
    return tmp_model, config, describe, False


def op_tanh(model, config, issue, j):  #
    tmp_model = modify_activations(model, 'tanh')
    tmp_model = modify_initializer(model, k_initializer='he_uniform')
    #tmp_model=modify_activations(model,relu)
    #because of the nan weight, in this step ,the weight of the last layer may still be nan even we changed the activations in the previous layers
    # this may lead to nan loss after repair. **reload the model will avoid this**.
    describe = "Using 'tanh' activation in each layers' activation; Use 'he_uniform' as the kernel initializer."
    return tmp_model, config, describe, False


def op_bn(model, config, issue, j):  #m
    tmp_model = BN_network(model, incert_layer='dense')
    describe = "Using 'BatchNormalization' layers after each Dense layers in the model."
    return tmp_model, config, describe, False


def op_initial(model, config, issue, j):  #
    good_initializer = [
        'he_uniform', 'lecun_uniform', 'glorot_normal', 'glorot_uniform',
        'he_normal', 'lecun_normal'
    ]
    #no clear strategy now
    init_1 = np.random.choice(good_initializer, 1)[0]
    init_2 = np.random.choice(good_initializer, 1)[0]
    tmp_model = modify_initializer(model, init_1, init_2)
    describe = "Using '{}' initializer as each layers' kernel initializer;\
         Use '{}' initializer as each layers' bias initializer.".format(str(init_2),str(init_1))
    return tmp_model, config, describe, False


def op_selu(model,config,issue,j):#m
    tmp_model=modify_activations(model,'selu')
    tmp_model=modify_initializer(model,'lecun_uniform','lecun_uniform')
    #selu usually use the lecun initializer
    describe = "Using 'SeLU' activation in each layers' activations; Use 'lecun_uniform' as the kernel initializer."
    return tmp_model,config,describe,False


def op_leaky(model,config,issue,j):#m
    #advanced_list=['leaky_re_lu','elu','prelu','softmax','activation','thresholded_re_lu','re_lu']
    leaky_list=['LeakyReLU','ELU','PReLU','ThresholdedReLU']
    tmp_model=modify_activations(model,leaky_list[j],method='special')
    describe = "Using advanced activation '{}' instead of each layers' activations."
    return tmp_model,config,describe,False


def op_adam(model,config,issue,j):#m
    tmp_model=model
    if config['opt']=='Adam' or (config['opt'].__class__==getattr(O, 'Adam')) :
        return tmp_model,config,True
    config['opt']='Adam'
    describe = "Using 'Adam' optimizer, the parameter setting is default."
    return tmp_model,config,describe,False


def op_lr(model,config,issue,j):#m
    tmp_model=model
    #tmp_list=[0.01,0.001,0.1,0.0001]
    kwargs_list=[]
    if (isinstance(issue,str) and issue=='training_unstable')\
        or (isinstance(issue,list) and 'training_unstable' in issue):
        kwargs_list.append(10**(-j-1))
    elif (isinstance(issue,str) and issue=='training_not_converge')\
       or (isinstance(issue,list) and 'training_not_converge' in issue) :
        kwargs_list.append(10**(-j-1))
    config['opt']=modify_optimizer(config['opt'],kwargs_list,method='lr')
    describe = "Using '{}' learning rate in the optimizer.".format(str(kwargs_list[0]))
    return tmp_model,config,describe,False


def op_ReduceLR(model,config,issue,j):#m
    tmp_model=model
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                patience=5, min_lr=0.001)
    if len(config['callbacks'])!=0:
        for call in range(len(config['callbacks'])):
            if config['callbacks'][call].__class__==reduce_lr.__class__:
                return tmp_model,config,True
    else:
        config['callbacks'].append(reduce_lr)
    describe = "Using 'ReduceLROnPlateau' callbacks in training."
    return tmp_model,config,describe,False


def op_momentum(model,config,issue,j):#m
    tmp_model=model
    kwargs_list=random_kwargs_list(method='momentum')
    config['opt']=modify_optimizer('SGD',kwargs_list,method='momentum')
    describe="Using 'momentum {}' in SGD optimizer in the optimizer.".format(str(kwargs_list[0]))
    return tmp_model,config,describe,False


def op_batch(model,config,issue,j):#m
    tmp_model=model
    #a=[2,4,8,16]
    config['batch_size']=(2**(j+1))*config['batch_size']
    describe="Using 'batch_size {}' in model training.".format(str(config['batch_size']))
    return tmp_model,config,describe,False


def op_GN(model,config,issue,j):#m
    for i in range(min(len(model.layers),3)):
        if ('gaussian_noise' in model.layers[i].name) or model.layers[i].__class__==getattr(L, 'GaussianNoise'):
            return model,config,True
    tmp_model=Gaussian_Noise(model)
    describe="Using 'Gaussian_Noise' after the input layer."
    return tmp_model,config,describe,False


def op_optimizer(model,config,issue,j):# no 
    tmp_model=model
    optimizer_list=['SGD','Adam','Nadam','Adamax','RMSprop']
    tmp=0
    while (tmp==0):
        tmp_opt=np.random.choice(optimizer_list,1)[0]
        tmp=1
        if config['opt']==tmp_opt or (config['opt'].__class__==getattr(O, tmp_opt)):
            tmp=0
            optimizer_list.remove(tmp_opt)
    config['opt']=tmp_opt
    describe='Using {} optimizer in model training, the parameter setting is default.'.format(str(tmp_opt))
    return tmp_model,config,describe,False


def op_EarlyStop(model,config,issue,j):# m
    tmp_model=model
    early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=3, verbose=0, mode='auto',baseline=None, restore_best_weights=False)
    if len(config['callbacks'])!=0:
        for call in range(len(config['callbacks'])):
            if config['callbacks'][call].__class__==early_stopping.__class__:
                return tmp_model,config,True
    else:
        config['callbacks'].append(early_stopping)
    describe="Using 'EarlyStopping' callbacks in model training."
    return tmp_model,config,describe,False


def op_dropout(model,config,issue,j):
    tmp_model=Dropout_network(model,incert_layer='dense')
    describe="Using 'Dropout' layers after each Dense layer."
    return tmp_model,config,describe,False

def op_regular(model,config,issue,j):
    #regular_list=[l2,l1,l1_l2]
    tmp_model=modify_regularizer(model)
    describe="Using 'l2 regularizer' in each Dense layers."
    return tmp_model,config,describe,False

def repair_default(model,config,issue,j):
    print('Wrong setting')
    os._exit(0)
