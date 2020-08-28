import os
import sys
sys.path.append('../../data')
sys.path.append('../../utils')
from mnist import load_data,preprocess
from utils import *
import numpy as np
import keras
from keras.optimizers import SGD,Adam
import datetime
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Activation
from keras.regularizers import l2
from keras.models import Model

class Lenet_model:
    def __init__(self, input_shape, cls_num=10):
        self.name = 'LeNet-5'
        self.input_shape = input_shape
        self.cls_num = cls_num

        self.kernel_size = (5,5)
        self.weight_decay = 0.0001


    def build_model(self):
        input = Input(shape=self.input_shape)
        x = Conv2D(6, self.kernel_size, padding='valid', activation='relu', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.weight_decay))(input)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(16, self.kernel_size, padding='valid', activation='relu', kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.weight_decay))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(120, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(x)
        x = Dense(84, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(x)
        x = Dense(self.cls_num, kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay))(x)
        x = Activation('softmax')(x)
        lenet5 = Model(input, x)
        return lenet5

    def build_simple_model(self):
        input = Input(shape=self.input_shape)
        x = Conv2D(6, self.kernel_size, padding='valid', activation='relu')(input)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(16, self.kernel_size, padding='valid', activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(120, activation='relu')(x)
        x = Dense(84, activation='relu')(x)
        x = Dense(self.cls_num)(x)
        x = Activation('softmax')(x)
        lenet5 = Model(input, x)
        return lenet5
    
    def build_tmp_model(self):
        input = Input(shape=self.input_shape)
        x = Conv2D(6, self.kernel_size, padding='valid', activation='relu')(input)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(16, self.kernel_size, padding='valid', activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(120, activation='relu')(x)
        x = Dense(84, activation='relu')(x)
        x = Dense(self.cls_num)(x)
        x = Activation('softmax')(x)
        lenet5 = Model(input, x)
        return lenet5


if __name__ == '__main__':
    tmp=Lenet_model(input_shape=(28,28,1))
    model=tmp.build_tmp_model()
    save_model(model,'/data/zxy/DL_tools/DL_tools/models/seed_model/tmp.h5')
    model=load_model('/data/zxy/DL_tools/DL_tools/models/seed_model/tmp.h5')
    #config:
    labels=10
    (x, y), (x_val, y_val)=load_data()
    x=preprocess(x,'tensorflow')
    x_val=preprocess(x_val,'tensorflow')
    y = keras.utils.to_categorical(y, labels)
    y_val = keras.utils.to_categorical(y_val, labels)
    #x=x[:10000,:,:,:]
    #y=y[:10000,:]
    #x_val=x_val[:2000,:,:,:]
    #y_val=y_val[:2000,:]
    #opt='SGD'
    #opt=SGD(lr=0.1)
    opt='Adam'
    loss='categorical_crossentropy'
    dataset={}
    dataset['x']=x
    dataset['y']=y
    dataset['x_val']=x_val
    dataset['y_val']=y_val
    epoch=30
    batch_size=256
    log_dir='../../log/simple_lenet_tmp/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path=log_dir+'simple_lenet_tmp.csv'
    fig_name=log_dir+'simple_lenet_tmp.pdf'
    callbacks=[]

    model,history=model_train(model=model,optimizer=opt,loss=loss,dataset=dataset,iters=epoch,batch_size=batch_size,log_path=log_path,callbacks=callbacks,verb=1)
    gradient_list,layer_outputs,wts=gradient_test(model,dataset,batch_size)
    tmp_list=[]
    for i in range(len(gradient_list)):
        zeros=np.sum(gradient_list[i]==0)
        tmp_list.append(zeros/gradient_list[i].size)
    ave_g,_=average_gradient(gradient_list)
    issue_list=determine_issue(gradient_list,history,layer_outputs,model,threshold_low=1e-3,threshold_high=1e+3)
    #problem=gradient_issue(gradient_list)
    result_dic=read_csv(log_path,epoch)
    generate_fig(result_dic,fig_name)
    print('finish')

    # model.optimizer.get_config()