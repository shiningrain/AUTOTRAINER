import os
import sys
sys.path.append('.')
# from utils import has_NaN
import matplotlib.pyplot as plt
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

tmp_model_path='./tmp_models'
Insert_Layers=['dense','lstm','conv']

def has_NaN(output):
    output=np.array(output)
    result=(np.isnan(output).any() or np.isinf(output).any())
    return result

def reload_model(model,path=tmp_model_path):
    path=os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(path)
    model_name='model_{}.h5'.format(str(os.getpid()))
    path=os.path.join(path,model_name)
    model.save(path)
    model=load_model(path)
    os.remove(path)
    return model

def random_kwargs_list(method='gradient_clip',clipvalue=10):
    kwargs_list=[]
    if method=='gradient_clip':
        tmp_list=['clipvalue']
        op_type=np.random.choice(tmp_list,1)[0]
        # kwargs_list.append(op_type)
        kwargs_list.append('clipvalue')#clipnorm will lead to error in monitor
        # if op_type=='clipnorm':
        #     kwargs_list.append(round(np.random.uniform(float(1-clipnorm),float(1+clipnorm)),2))
        if op_type=='clipvalue':
            kwargs_list.append(round(np.random.uniform(float(clipvalue-1),clipvalue),2))
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

def modify_initializer(model,b_initializer=None,k_initializer=None):
    layers_num = len(model.layers)
    if isinstance(b_initializer,str):
        bias_initializer = getattr(I, b_initializer)()
    else:
        bias_initializer=b_initializer()
    if isinstance(k_initializer,str):
        kernel_initializer = getattr(I, k_initializer)()
    else:
        kernel_initializer=k_initializer()
    # bias_initializer = getattr(I, b_initializer)
    # kernel_initializer = getattr(I, k_initializer)
    last=last_layer(model.layers)
    for i in range(int(last)):#the last layer don't modify
        if ('lstm' not in model.layers[i].name):
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
    advanced_list=['leaky_re_lu','elu','softmax','activation','thresholded_re_lu','re_lu']#,'prelu' lead to gradient_message error
    for j in range(len(advanced_list)):
        if (i+1)<len(model.layers) and advanced_list[j] in model.layers[i+1].get_config()['name']:
            if model.layers[i].get_config()['activation']!=linear:
                model.layers[i].activation=linear
            return False
        if advanced_list[j] in model.layers[i].get_config()['name']:
            return False
    return True


def modify_activations(model,activation_name,method='normal'):#https://github.com/keras-team/keras/issues/9370
    """
    normal method: activaiton is a function
    special method activatio is a string
    """
    lstm_layer=['lstm','rnn','gru']
    layers_num = len(model.layers)
    if method=='normal':# For relu
        if isinstance(activation_name,str):
            activation=getattr(A, activation_name)
        last=last_layer(model.layers)
        for i in range(int(last)):###考虑special情况
            lstm_judge=False
            for j in lstm_layer:
                if j in model.layers[i].name:
                    lstm_judge=True
            if ('activation' in model.layers[i].get_config()) and\
                not_dense_acti(model,i):
                if (lstm_judge==True and not (activation_name=='relu' or activation_name=='selu')):
                    new_config=copy.deepcopy(model.layers[i].get_config())
                    new_config['activation']=activation
                    new_layer=model.layers[i].__class__(**new_config)
                    model=replace_intermediate_layer_in_keras(model,i,new_layer)
                elif (lstm_judge==True and (activation_name=='relu' or activation_name=='selu')):
                    new_config=copy.deepcopy(model.layers[i].get_config())
                    new_config['activation']=getattr(A, 'tanh')
                    new_layer=model.layers[i].__class__(**new_config)
                    model=replace_intermediate_layer_in_keras(model,i,new_layer)
                else:
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
                    lstm_judge=False
                    for j in lstm_layer:
                        if j in model.layers[i].name:
                            lstm_judge=True
                    if lstm_judge==True:
                        i+=1
                    else:
                        model.layers[i].activation=linear
                        act_cls = getattr(L, activation_name)
                        model = insert_intermediate_layer_in_keras(model,i+1,act_cls())
                        i+=1
                        layers_num+=1
            i+=1
        model.summary()
    for i in range(len(model.layers)):
        for j in range(len(model.layers[i].get_weights())):
            if has_NaN(model.layers[i].get_weights()[j]):
                new_config=copy.deepcopy(model.layers[i].get_config())
                new_layer=model.layers[i].__class__(**new_config)
                model=replace_intermediate_layer_in_keras(model,i,new_layer)
                break
    model=reload_model(model)
    return model

def modify_regularizer(model,kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)):
    last=last_layer(model.layers)
    lstm_layer=['lstm','rnn','gru']
    break_count=0
    for i in range(int(last)):#the last layer don't modify
        if break_count>=2:
            return model, True
        lstm_judge=False
        for j in lstm_layer:
            if j in model.layers[i].name:
                lstm_judge=True
        if lstm_judge==False:
            if 'kernel_regularizer' in model.layers[i].get_config():
                if model.layers[i].kernel_regularizer!=None and model.layers[i].kernel_regularizer.get_config()['l2']>=0.7*0.01:
                    break_count+=1
                    continue    
                model.layers[i].kernel_regularizer=kernel_regularizer
            if 'bias_regularizer' in model.layers[i].get_config():
                model.layers[i].kernel_regularizer=bias_regularizer
    model=reload_model(model)
    return model,False

def Dropout_network(model,incert_layer=Insert_Layers,rate=0.25):
    layers_num = len(model.layers)
    drop_count=0
    max_drop=int(layers_num/4)
    i=0
    while(i<layers_num-1):#the last layer don't add BN layer
        for j in incert_layer:
            if j in model.layers[i].get_config()['name']:
                if model.layers[i+1].__class__!=getattr(L,'Dropout'):
                    model = insert_intermediate_layer_in_keras(model,i+1,Dropout(rate=rate,name='Drop_{}'.format(i)))
                    drop_count+=1
                    i+=1
                    if drop_count>=max_drop:
                        i=layers_num
                    layers_num+=1
        i+=1
    model.summary()
    model=reload_model(model)
    return model

def BN_network(model,incert_layer=Insert_Layers):
    layers_num = len(model.layers)
    i=0
    while(i<layers_num-1):#the last layer don't add BN layer
        for j in incert_layer:
            if j in model.layers[i].get_config()['name']:
                if model.layers[i+1].__class__!=getattr(L,'BatchNormalization'):
                    model = insert_intermediate_layer_in_keras(model,i+1,BatchNormalization())
                    i+=1
                    layers_num+=1
        i+=1
        # break
    model.summary()
    model=reload_model(model)
    return model


def Gaussian_Noise(model,stddev=0.1):
    start_layers=['conv','lstm','rnn','gru','dense']
    for i in range(len(model.layers)):
        for j in range(len(start_layers)):
            if start_layers[j] in model.layers[i].name:
                model = insert_intermediate_layer_in_keras(model,i,GaussianNoise(stddev))
                model=reload_model(model)
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

def modify_optimizer(optimizer,kwargs_list,method='lr'):
    if isinstance(optimizer,str):
        opt_cls = getattr(O, optimizer)
        optimizer = opt_cls()
    if method=='lr':
        current_lr=K.eval(optimizer.lr)
        kwargs=optimizer.get_config()
        kwargs['lr']=kwargs_list[0]*current_lr
        new_opt=optimizer.__class__(**kwargs)
    elif method=='momentum':
        new_opt=SGD(momentum=kwargs_list[0])
    elif method=='gradient':
        # add gradient clip or gradient norm, kwargs list contains a optimizer name and its kwargs now.
        kwargs=optimizer.get_config()
        kwargs[kwargs_list[0]]=kwargs_list[-1]
        new_opt=optimizer.__class__(**kwargs)
    return new_opt

def repair_strategy(method='balance'):
    if method=='balance':
        #first order: efficiency - complexity:
        #second order: efficiency, last order: complexity.
        gradient_vanish_strategy=['selu_1','relu_1','bn_1']#'selu_1',
        gradient_explode_strategy=['selu_1','relu_1','gradient_2','tanh_1','bn_1']#
        dying_relu_strategy=['selu_1','bn_1','initial_3','leaky_3']#
        unstable_strategy=['adam_1','lr_3','ReduceLR_1','batch_4','momentum_3','GN_1','initial_3']#
        not_converge_strategy=['optimizer_3','lr_3','initial_3']
        over_fitting_strategy=['regular_1','estop_1','dropout_1','GN_1']#
    elif method=='structure':
        #first order: complexity; seconde order: efficiency.
        gradient_vanish_strategy=['relu_1','selu_1','bn_1']
        gradient_explode_strategy=['gradient_2','relu_1','selu_1','tanh_1','bn_1']#1 ,
        dying_relu_strategy=['selu_1','initial_3','leaky_3','bn_1']#
        unstable_strategy=['adam_1','lr_3','ReduceLR_1','batch_4','momentum_3','GN_1','initial_3']#
        not_converge_strategy=['optimizer_3','lr_3','initial_3']
        over_fitting_strategy=['estop_1','regular_1','dropout_1','GN_1']
        # over_fitting_strategy=['regular_1','dropout_1','GN_1']
    elif method=='efficiency':
        #first order: efficiency; seconde order: complexity.
        gradient_vanish_strategy=['relu_1','selu_1','bn_1']
        gradient_explode_strategy=['relu_1','selu_1','tanh_1','gradient_2','bn_1']#1 ,
        dying_relu_strategy=['selu_1','initial_3','leaky_3','bn_1']#
        unstable_strategy=['adam_1','lr_3','ReduceLR_1','batch_4','momentum_3','GN_1','initial_3']#
        not_converge_strategy=['optimizer_3','lr_3','initial_3']
        over_fitting_strategy=['regular_1','dropout_1','estop_1','GN_1']
        # over_fitting_strategy=['regular_1','dropout_1','GN_1']
    else:
        print('Not support this method')
        os._exit(0)#you can design your repair strategy here.
    return [gradient_vanish_strategy,gradient_explode_strategy,dying_relu_strategy,unstable_strategy,not_converge_strategy,over_fitting_strategy]

##------------------------add solution describe here-----------------------------
def op_gradient(model, config, issue, j,config_set):  #m
    describe=0
    tmp_model = model
    if isinstance(config['opt'],str):
        opt_cls = getattr(O, config['opt'])
        config['opt'] = opt_cls()
    if ('clipvalue'
            in config['opt'].get_config()) or ('clipnorm'
                                               in config['opt'].get_config()):
        return tmp_model, config,describe, True,config_set
    kwargs_list = random_kwargs_list(method='gradient_clip')
    config['opt'] = modify_optimizer(config['opt'],
                                     kwargs_list,
                                     method='gradient')
    config_set['opt_kwargs'][kwargs_list[0]]=kwargs_list[-1]
    describe = "Using 'Gradient Clip' operation, add {}={} to the optimizer".format(
        str(kwargs_list[0]), str(kwargs_list[-1]))
    return tmp_model, config,describe, False,config_set


def op_relu(model, config, issue, j,config_set):  #
    tmp_model = modify_initializer(model, k_initializer='he_uniform',b_initializer='zeros')
    tmp_model = modify_activations(tmp_model, 'relu')
    describe = "Using 'ReLU' activation in each layers' activations; Use 'he_uniform' as the kernel initializer."
    return tmp_model, config, describe, False,config_set


def op_tanh(model, config, issue, j,config_set):  #
    tmp_model = modify_activations(model, 'tanh')
    tmp_model = modify_initializer(tmp_model, k_initializer='he_uniform',b_initializer='zeros')
    describe = "Using 'tanh' activation in each layers' activation; Use 'he_uniform' as the kernel initializer."
    return tmp_model, config, describe, False,config_set


def op_bn(model, config, issue, j,config_set):  #m
    tmp_model = BN_network(model)
    describe = "Using 'BatchNormalization' layers after each Dense layers in the model."
    return tmp_model, config, describe, False,config_set


def op_initial(model, config, issue, j,config_set):  #
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
    return tmp_model, config, describe, False,config_set


def op_selu(model,config,issue,j,config_set):#m
    tmp_model=modify_activations(model,'selu')
    tmp_model=modify_initializer(tmp_model,'lecun_uniform','lecun_uniform')
    #selu usually use the lecun initializer
    describe = "Using 'SeLU' activation in each layers' activations; Use 'lecun_uniform' as the kernel initializer."
    return tmp_model,config,describe,False,config_set


def op_leaky(model,config,issue,j,config_set):#m
    leaky_list=['LeakyReLU','ELU','ThresholdedReLU']
    tmp_model=modify_activations(model,leaky_list[j],method='special')
    describe = "Using advanced activation '{}' instead of each layers' activations."
    return tmp_model,config,describe,False,config_set


def op_adam(model,config,issue,j,config_set):#m
    describe=0
    tmp_model=model
    if config['opt']=='Adam' or (config['opt'].__class__==getattr(O, 'Adam')) :
        return tmp_model,config,describe,True,config_set
    config['opt']='Adam'
    config_set['optimizer']='Adam'
    config_set['opt_kwargs']={}
    describe = "Using 'Adam' optimizer, the parameter setting is default."
    return tmp_model,config,describe,False,config_set


def op_lr(model,config,issue,j,config_set):#m
    tmp_model=model
    kwargs_list=[]
    describe=0
    
    if config['opt'].__class__==getattr(O, 'SGD'):
        if issue=='not_converge':    
            lr_try=0.01*(10**(j))
        else:
            lr_try=0.01*(0.1**(j))
        if K.eval(config['opt'].lr)!=lr_try:
            kwargs_list.append(lr_try/K.eval(config['opt'].lr))
        else:
            return model,config,describe,True,config_set
    else:
        if issue=='not_converge':    
            lr_try=0.001*(10**(j))
        else:
            lr_try=0.001*(0.1**(j))
        if K.eval(config['opt'].lr)!=lr_try:
            kwargs_list.append(lr_try/K.eval(config['opt'].lr))
        else:
            return model,config,describe,True,config_set

    config['opt']=modify_optimizer(config['opt'],kwargs_list,method='lr')
    config_set['opt_kwargs']['lr']=K.eval(config['opt'].lr)
    describe = "Using '{}' learning rate in the optimizer.".format(str(kwargs_list[0]))
    return tmp_model,config,describe,False,config_set


def op_ReduceLR(model,config,issue,j,config_set):#m
    describe=0
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                patience=5, min_lr=0.001)
    if len(config['callbacks'])!=0:
        for call in range(len(config['callbacks'])):
            if config['callbacks'][call].__class__==reduce_lr.__class__:
                return model,config,describe,True,config_set
    else:
        config['callbacks'].append(reduce_lr)
    if 'callbacks' not in config_set.keys():
        config_set['callbacks']=['ReduceLR']
    else:
        config_set['callbacks'].append('ReduceLR')
    describe = "Using 'ReduceLROnPlateau' callbacks in training."
    return model,config,describe,False,config_set


def op_momentum(model,config,issue,j,config_set):#m
    describe=0
    tmp_model=model
    kwargs_list=random_kwargs_list(method='momentum')
    config['opt']=modify_optimizer('SGD',kwargs_list,method='momentum')
    config_set['optimizer']='SGD'
    config_set['opt_kwargs']={}
    config_set['opt_kwargs']['momentum']=kwargs_list[0]
    describe="Using 'momentum {}' in SGD optimizer in the optimizer.".format(str(kwargs_list[0]))
    return tmp_model,config,describe,False,config_set


# def op_batch(model,config,issue,j,config_set):#m
#     tmp_model=model
#     #a=[2,4,8,16]
#     config['batch_size']=(2**(j+1))*config['batch_size']
#     describe="Using 'batch_size {}' in model training.".format(str(config['batch_size']))
#     config_set['batchsize']=config['batch_size']
#     return tmp_model,config,describe,False,config_set

def op_batch(model,config,issue,j,config_set):#m
    tmp_model=model
    batch_try=(32*(2**j))
    describe=0
    if config['batch_size']!=batch_try:
        config['batch_size']=batch_try
    else: 
        return model,config,describe,True,config_set
    describe="Using 'batch_size {}' in model training.".format(str(config['batch_size']))
    config_set['batchsize']=config['batch_size']
    return tmp_model,config,describe,False,config_set


def op_GN(model,config,issue,j,config_set):#m
    describe=0
    for i in range(min(len(model.layers),3)):
        if ('gaussian_noise' in model.layers[i].name) or model.layers[i].__class__==getattr(L, 'GaussianNoise'):
            return model,config,describe,True,config_set
    tmp_model=Gaussian_Noise(model)
    describe="Using 'Gaussian_Noise' after the input layer."
    return tmp_model,config,describe,False,config_set


def op_optimizer(model,config,issue,j,config_set):# no 
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
    config_set['optimizer']=tmp_opt
    config_set['opt_kwargs']={}
    describe='Using {} optimizer in model training, the parameter setting is default.'.format(str(tmp_opt))
    return tmp_model,config,describe,False,config_set


def op_EarlyStop(model,config,issue,j,config_set):# m
    describe=0
    patience=max(3,int(config['epoch']/15))
    early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=patience, verbose=0, mode='auto',baseline=None, restore_best_weights=False)
    if len(config['callbacks'])!=0:
        for call in range(len(config['callbacks'])):
            if config['callbacks'][call].__class__==early_stopping.__class__:
                return model,config,describe,True,config_set
    config['callbacks'].append(early_stopping)
    if 'callbacks' not in config_set.keys():
        config_set['callbacks']=['estop']
    else:
        config_set['callbacks'].append('estop')
    describe="Using 'EarlyStopping' callbacks in model training."
    return model,config,describe,False,config_set


def op_dropout(model,config,issue,j,config_set):
    tmp_model=Dropout_network(model)
    describe="Using 'Dropout' layers after each Dense layer."
    return tmp_model,config,describe,False,config_set

def op_regular(model,config,issue,j,config_set):
    #regular_list=[l2,l1,l1_l2]
    tmp_model,judge=modify_regularizer(model)
    describe="Using 'l2 regularizer' in each Dense layers."
    if judge==True:
        return model,config,describe,True,config_set
    return tmp_model,config,describe,False,config_set

def repair_default(model,config,issue,j):
    print('Wrong setting')
    os._exit(0)
