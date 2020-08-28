import os
import sys
sys.path.append('../../data')
sys.path.append('../../utils')
sys.path.append('../../repair')
#from cifar10 import load_data,preprocess
from utils import *
from repair import *
import numpy as np
import keras
from keras.optimizers import SGD,Adam
from keras.models import load_model
import datetime
import numpy as np
from matplotlib import pyplot
from keras.datasets import cifar10
import keras.optimizers as O
import argparse
import pickle
import itertools
import importlib
import keras.backend as K

callbacks=[]
labels=10

def get_dataset(dataset,method='multi'):
    data_name=dataset.split('_')[0]
    data = importlib.import_module('{}'.format(data_name.lower()), package='data')
    if data_name=='simplednn':
        choice=dataset.split('_')[-1]
        (x, y), (x_val, y_val) = data.load_data(method=choice)
    else:(x, y), (x_val, y_val) = data.load_data()
    preprocess_func = data.preprocess
    dataset={}
    dataset['x']=preprocess_func(x)
    dataset['x_val']=preprocess_func(x_val)
    if method=='multi':
        dataset['y']=keras.utils.to_categorical(y, labels)
        dataset['y_val']=keras.utils.to_categorical(y_val, labels)
    else:
        dataset['y']=y
        dataset['y_val']=y_val
    return dataset

def get_feature_string(strings):
    tmp_list=[]
    for i in range(len(strings)):
        file_name=strings[i].split('/')[-1]
        delete=file_name.split('-')[0]
        feature=file_name.replace(delete,'')
        tmp_list.append(feature.split('.')[0])
    string=str(tmp_list[0])
    for j in range(1,len(tmp_list)):
        string=string+tmp_list[j]
    return string


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Generation')
    parser.add_argument('--model_dir','-md',default='/data/zxy/DL_tools/DL_tools/code/AutoGeneration/simplednn/model', help='model dir')
    parser.add_argument('--config_dir', '-cd',default='/data/zxy/DL_tools/DL_tools/code/AutoGeneration/simplednn/config', help='The dir to store configurations')
    parser.add_argument('--log_dir', '-ld', type=str, default='/data/zxy/DL_tools/DL_tools/code/AutoGeneration/simplednn/log', help='The path to save model')
    args = parser.parse_args()

    model_path_list=[]
    config_path_list=[]

    for root,dirs,files in os.walk(args.model_dir):
        for eachfile in files:
            if (eachfile.split('.')[-1]=='h5'):
                tmp_path=root+'/'+eachfile
                model_path_list.append(tmp_path)
    for root,dirs,files in os.walk(args.config_dir):
        for eachfile in files:
            if (eachfile.split('.')[-1]=='pkl'):
                tmp_path=root+'/'+eachfile
                config_path_list.append(tmp_path)
    combine=list(itertools.product(model_path_list[:5],config_path_list[:3]))
    
    for i in range(len(combine)):
        model_path=combine[i][0]
        model=load_model(combine[i][0])
        with open(combine[i][1], 'rb') as f:
            training_config= pickle.load(f)
        
        opt_cls = getattr(O, training_config['optimizer'])
        opt = opt_cls(**training_config['opt_kwargs'])
        modify_optimizer(opt,[0.1],method='lr')
        batch_size=training_config['batchsize']
        epoch=training_config['epoch']
        loss=training_config['loss']
        dataset=get_dataset(training_config['dataset'],method='single')
        #dataset=get_dataset(training_config['dataset'])

        feature_string=get_feature_string(combine[i])
        log_dir=os.path.join(args.log_dir,feature_string)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path=os.path.join(log_dir,'training.csv')
        fig_name=os.path.join(log_dir,'training.pdf')

        model.summary()
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
        print(1)
        

    