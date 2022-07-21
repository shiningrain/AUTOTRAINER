import os
import argparse
import sys

#from utils import model_train

sys.setrecursionlimit(1000000)
import uuid
sys.path.append('./data')
sys.path.append('./utils')
from utils import *
from modules import *
import numpy as np
import keras
import keras.optimizers as O
from keras.models import load_model
import argparse
import pickle
import itertools
import importlib
import keras.backend as K

def get_dataset():
    # data_name=dataset.split('_')[0]
    # data = importlib.import_module('{}'.format(data_name.lower()), package='data')
    # if data_name=='simplednn':
    #     choice=dataset.split('_')[-1]
    #     (x, y), (x_val, y_val) = data.load_data(method=choice)
    # else:
    #     (x, y), (x_val, y_val) = data.load_data()
    # preprocess_func = data.preprocess
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    dataset={}
    dataset['x']=x_train
    dataset['x_val']=x_test
    dataset['y']=y_train
    dataset['y_val']=y_test
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OL Problem Demo Detection & Autorepair')
    #parser.add_argument('--model_path','-mp',default='./new_model.h5', help='model path')
    parser.add_argument('--config_path', '-cp',default='/data/zxy/DL_tools/DL_tools/AUTOTRAINER/AutoTrainer/demo_case/config_1.pkl', help='training configurations path')
    parser.add_argument('--check_interval', '-ci',default=3, help='detection interval')
    parser.add_argument('--result_dir', '-rd',default='./tmp1/result_dir', help='The dir to store results')
    parser.add_argument('--log_dir', '-ld',default='./tmp1/log_dir', help='The dir to store logs')
    parser.add_argument('--new_issue_dir', '-nd',default='./tmp1/new_issue', help='The dir to store models with new problem in detection')
    parser.add_argument('--root_dir', '-rtd',default='./tmp1', help='The root dir for other records')
    args = parser.parse_args()

    #model = load_model(os.path.abspath(args.model_path))
    # model=load_model('new_model.h5')
    # model.summary()
    count=0
    for i in range(50):
        import shutil
        shutil.rmtree(args.root_dir)
        print('===={}===='.format(i))
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.8))
        model.add(keras.layers.Dense(10, activation="relu"))

        # model.get_weights()
        # model.summary()

        with open(args.config_path, 'rb') as f:#input,bug type,params
            training_config = pickle.load(f)

        # print(training_config)

        opt_cls = getattr(O, training_config['optimizer'])
        opt = opt_cls(**training_config['opt_kwargs'])
        # print(opt)
        #opt ="adam"
        batch_size=training_config['batchsize']
        #batch_size = 128
        # print(batch_size)
        epoch=training_config['epoch']
        #epoch = 15
        loss=training_config['loss']
        # print(loss)
        #loss="binary_crossentropy"
        #dataset=get_dataset(training_config['dataset'])
        dataset = get_dataset()
        # print(dataset)
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
                    'Theta': 0.6
                    }


        train_result,_,_=model_train(model=model,train_config_set=training_config,optimizer=opt,loss=loss,dataset=dataset,iters=epoch,batch_size=batch_size,\
                    callbacks=callbacks,verb=1,checktype=check_interval,autorepair=False,save_dir=save_dir,determine_threshold=1,params=params,log_dir=log_dir,\
                    new_issue_dir=new_issue_dir,root_path=root_path)
        if train_result==1:
            count+=1
    print(1)
