# data issue, 2 bugs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import sys
sys.setrecursionlimit(1000000)
import uuid
sys.path.append('../../data')
sys.path.append('../../utils')
from utils import *
from modules import *
import numpy as np
import tensorflow as keras
import tensorflow.keras.optimizers as O
from tensorflow.keras.models import load_model
import argparse
import pickle
import itertools
import importlib
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.utils import to_categorical

def get_dataset(dataset):
    data_name=dataset.split('_')[0]
    data = importlib.import_module('{}'.format(data_name.lower()), package='data')
    if data_name=='simplednn':
        choice=dataset.split('_')[-1]
        (x, y), (x_val, y_val) = data.load_data(method=choice)
    else:
        (x, y), (x_val, y_val) = data.load_data()
    preprocess_func = data.preprocess
    dataset={}
    # dataset['x']=preprocess_func(x)
    # dataset['x_val']=preprocess_func(x_val)
    dataset['x']=x
    dataset['x_val']=x_val
    dataset['y']=to_categorical(y, 10)
    dataset['y_val']=to_categorical(y_val, 10)
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OL Problem Demo Detection & Autorepair')
    parser.add_argument('--model_path','-mp',default='./model_data.h5', help='model path')
    parser.add_argument('--config_path', '-cp',default='./config_data.pkl', help='training configurations path') 
    parser.add_argument('--check_interval', '-ci',default=3, help='detection interval') 
    parser.add_argument('--result_dir', '-rd',default='./tmp_data/result_dir', help='The dir to store results') 
    parser.add_argument('--log_dir', '-ld',default='./tmp_data/log_dir', help='The dir to store logs') 
    parser.add_argument('--new_issue_dir', '-nd',default='./tmp_data/new_issue', help='The dir to store models with new problem in detection') 
    parser.add_argument('--root_dir', '-rtd',default='./tmp_data', help='The root dir for other records') 
    args = parser.parse_args()
    
    if os.path.exists(args.root_dir):
        import shutil
        shutil.rmtree(args.root_dir)
    
    # #Initialize
    model = Sequential()
    #init = RandomUniform(minval=0, maxval=1)
    init='he_uniform'
    model.add(Conv2D(filters = 16,
            kernel_size = (3, 3),
            padding = 'valid',
            input_shape = (32, 32, 3),
            activation = 'relu',kernel_initializer=init))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    #建立第二个卷积层， filters 卷积核个数 36 个，kernel_size 卷积核大小 3*3
    #padding 是否零填充 valid 表示填充， activation 激活函数 relu
    model.add(Conv2D(filters = 36,
                    kernel_size = (3, 3),
                    padding = 'valid',
                    activation='relu',kernel_initializer=init))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    #建立平坦层，将多维向量转化为一维向量
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu',kernel_initializer=init))
    #建立隐藏层，隐藏层有 128 个神经元， activation 激活函数用 relu
    model.add(Dense(128, activation = 'relu',kernel_initializer=init))#,
    #model.add(BatchNormalization())
    #加入Dropout避免过度拟合
    model.add(Dropout(0.25))
    #建立输出层，一共有 10 个神经元，因为 0 到 9 一共有 10 个类别， activation 激活函数用 softmax 这个函数用来分类
    model.add(Dense(10, activation = 'softmax')) 
    model.save(args.model_path)
    training_config={}
    training_config['optimizer']='Adam'
    training_config['opt_kwargs']={'lr': 0.001}
    training_config['batchsize']=32
    training_config['epoch']=15
    training_config['dataset']='cifar10'
    training_config['loss']='categorical_crossentropy'#'mean_squared_error'
    with open(args.config_path, 'wb') as f:
        pickle.dump(training_config, f)
    print(1)

    model=load_model(args.model_path)

    with open(args.config_path, 'rb') as f:#input,bug type,params
        training_config = pickle.load(f)
    opt_cls = getattr(O, training_config['optimizer'])
    opt = opt_cls(**training_config['opt_kwargs'])
    batch_size=training_config['batchsize']
    epoch=training_config['epoch']
    loss=training_config['loss']
    dataset=get_dataset(training_config['dataset'])
    if 'callbacks' not in training_config.keys():
        callbacks=[]
    else:
        callbacks=training_config['callbacks']
    check_interval='epoch_'+str(args.check_interval)

    save_dir=args.result_dir
    log_dir=args.log_dir
    new_issue_dir=args.new_issue_dir
    root_path=args.root_dir
    
    params = {'beta_1': 1e-3,
                 'beta_2': 1e-4,
                 'beta_3': 70,
                 'gamma': 0.7,
                 'zeta': 0.03,
                 'eta': 0.2,
                 'delta': 0.01,
                 'alpha_1': 0,
                 'alpha_2': 0,
                 'alpha_3': 0,
                 'Theta': 0.7,
                 'omega_1':10,
                 'omega_2':1
                 }


    train_result,_,_=model_train(model=model,train_config_set=training_config,optimizer=opt,loss=loss,dataset=dataset,iters=epoch,batch_size=batch_size,\
                callbacks=callbacks,verb=1,checktype=check_interval,autorepair=True,save_dir=save_dir,determine_threshold=1,params=params,log_dir=log_dir,\
                new_issue_dir=new_issue_dir,root_path=root_path)