'''
··  Author: your name
Date: 2020-08-31 15:32:46
LastEditTime: 2021-12-28 20:31:52
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /Gradient_Vanish_Case/demo.py
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import sys
sys.setrecursionlimit(1000000)
import uuid
sys.path.append('./data')
sys.path.append('./utils')
from utils import *
from modules import *
import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.optimizers as O
from tensorflow.keras.models import load_model
import argparse
import pickle
import itertools
import importlib

def get_dataset(dataset_name):
    preprocess=True
    data_name=dataset_name.split('_')[0]
    if 'nopre' in dataset_name:
        dataset_name=dataset_name.split('-')[0]
        preprocess=False
    data = importlib.import_module('{}'.format(data_name.split('-').lower()), package='data')
    if data_name=='simplednn':
        choice=dataset_name.split('_')[-1]
        (x, y), (x_val, y_val) = data.load_data(method=choice)
    else:
        (x, y), (x_val, y_val) = data.load_data()
    preprocess_func = data.preprocess

    dataset={}
    if preprocess:
        dataset['x']=preprocess_func(x)
        dataset['x_val']=preprocess_func(x_val)
    else:
        dataset['x'],dataset['x_val']=x,x_val
    dataset['y']=y
    dataset['y_val']=y_val
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OL Problem Demo Detection & Autorepair')
    parser.add_argument('--model_path','-mp',default='/data/zxy/DL_tools/DL_tools/code/AutoGeneration/benchmark/simplednn/2022_combine/1a46f5fa-e5a8-336f-9f62-ec31286b0cb2/model_al_ci_10.h5', help='model path')
    parser.add_argument('--config_path', '-cp',default='/data/zxy/DL_tools/DL_tools/code/AutoGeneration/benchmark/simplednn/2022_combine/1a46f5fa-e5a8-336f-9f62-ec31286b0cb2/config__al_ci_.pkl', help='training configurations path') 
    parser.add_argument('--check_interval', '-ci',default=3, help='detection interval') 
    parser.add_argument('--result_dir', '-rd',default='./tmp/result_dir', help='The dir to store results') 
    parser.add_argument('--log_dir', '-ld',default='./tmp/log_dir', help='The dir to store logs') 
    parser.add_argument('--new_issue_dir', '-nd',default='./tmp/new_issue', help='The dir to store models with new problem in detection') 
    parser.add_argument('--root_dir', '-rtd',default='./tmp', help='The root dir for other records') 
    args = parser.parse_args()

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

    # model=load_model('/data/zxy/DL_tools/DL_tools/AUTOTRAINER/AutoTrainer/demo_case/Gradient_Vanish_Case/model.h5')
    train_result,_,_=model_train(model=model,train_config_set=training_config,optimizer=opt,loss=loss,dataset=dataset,iters=epoch,batch_size=batch_size,\
                callbacks=callbacks,verb=1,checktype=check_interval,autorepair=True,save_dir=save_dir,determine_threshold=1,params=params,log_dir=log_dir,\
                new_issue_dir=new_issue_dir,root_path=root_path)
            