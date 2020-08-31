import os
import sys
import psutil
sys.path.append('.')
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
logger = Logger()
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import monitor as mn
import pickle
import time
import modules as md

default_param = {'beta_1': 1e-3,
                 'beta_2': 1e-4,
                 'beta_3': 70,
                 'gamma': 0.7,
                 'zeta': 0.03,
                 'eta': 0.2,
                 'delta': 0.01,
                 'alpha_1': 0,
                 'alpha_2': 0,
                 'alpha_3': 0,
                 'Theta': 0.7
                 }


class LossHistory(keras.callbacks.Callback):

    def __init__(self,training_data,model,batch_size,total_epoch,save_dir,determine_threshold=5,satisfied_acc=0.7,\
        checktype='epoch_5',satisfied_count=3,retrain=False,pkl_dir=None,solution=None,params={}): #only support epoch method now
        """[summary]

        Args:
            training_data ([list]): [training dataset]
            model ([model]): [untrained model]
            batch_size ([int]): [batch size]
            save-dir([str]):[the dir to save the detect result]
            checktype (str, optional): [checktype,'a_b', a can be chosen from ['epoch', 'batch'], b is number, it means the monitor will check \
            the gradient and loss every 'b' 'a'.]. Defaults to 'epoch_5'.
            satisfied_acc (float, optional): [the satisfied accuracy, when val accuracy beyond this, the count ++, when the count is bigger or\
                equal to satisfied_count, training stop.]. Defaults to 0.7.
            satisfied_count (int, optional): []. Defaults to 3.

        """
        self.trainX = training_data[0]
        self.trainy = training_data[1]
        self.batch_size=batch_size
        self.model=model
        self.satisfied_acc=satisfied_acc
        self.satisfied_count=satisfied_count
        self.count=0
        self.checktype=checktype.split('_')[0]
        self.checkgap=int(checktype.split('_')[-1])
        self.issue_list=[]
        self.save_dir=save_dir
        if not os.path.exists(save_dir):# record monitor and repair message
            os.makedirs(save_dir)
        self.pkl_dir=pkl_dir
        self.retrain=retrain
        self.total_epoch=total_epoch
        self.determine_threshold=determine_threshold
        self.params=params
        if self.params=={}:
            self.params=default_param

        self.history={}
        self.history['loss']=[]
        self.history['acc']=[]
        self.history['val_loss']=[]
        self.history['val_acc']=[]

        self.start_time=time.time()
        self.log_name='{}_{}.log'.format('monitor','detection')
        if self.retrain==True:
            self.log_name='{}_{}.log'.format('monitor','repair')
            self.solution=solution
        self.log_name=os.path.join(self.save_dir,self.log_name)

        if os.path.exists(self.log_name):
            os.remove(self.log_name)## avoid the repeat writing

        self.log_file=open(self.log_name,'a+')
        self.log_file.write('{},{},{},{},{}\n'.format('checktype','current_epoch','issue_list','time_usage','Describe'))

        self.Monitor=mn.IssueMonitor(total_epoch,self.satisfied_acc,self.params,self.determine_threshold)

    def on_train_begin(self,logs=None):
        weights=self.model.trainable_weights# get trainable weights
        grads = self.model.optimizer.get_gradients(self.model.total_loss, weights)
        symb_inputs = [self.model._feed_inputs , self.model._feed_targets , self.model._feed_sample_weights,K.learning_phase()]#input,corresponding label,weight of each sample(all of them are 1),learning rate(we set it to 0)
        self.f = K.function(symb_inputs, grads)
        if self.retrain==True:
            self.log_file.write('-----Using {} solution to retrain Detail can be found in the directory!-----\n'.format(self.solution))

    def on_epoch_end(self,epoch,logs={}):
        self.history['loss'].append(logs.get('loss'))
        self.history['acc'].append(logs.get('accuracy'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_acc'].append(logs.get('val_accuracy'))
        if (epoch)%self.checkgap==0:
            
            trainingExample = self.trainX[0:self.batch_size,...]
            trainingY=self.trainy[0:self.batch_size]
            x, y, sample_weight = self.model._standardize_user_data(trainingExample, trainingY)
            #output_grad = f(x + y + sample_weight)
            self.evaluated_gradients = self.f([x , y , sample_weight,0])
            gradient_list=[]
            for i in range(len(self.evaluated_gradients)):
                if isinstance(self.evaluated_gradients[i],np.ndarray):
                    gradient_list.append(self.evaluated_gradients[i])

            self.issue_list=self.Monitor.determine(self.model,self.history,gradient_list,self.checkgap)
            self.issue_list=md.filtered_issue(self.issue_list)

            self.evaluated_gradients=0
            gradient_list=0

            if self.retrain==False:
                if self.issue_list==[]:
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype,epoch,self.issue_list,\
                        str(time.time()-self.start_time),'No Issue now'))
                    self.log_file.flush()
                elif self.issue_list==['need_train']:
                    self.issue_list=list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype,epoch,self.issue_list,\
                        str(time.time()-self.start_time),'NO training problems now. You need to train this new model more times.'))
                    self.log_file.flush()
                    print('NO training problems now. You need to train this new model more iterations.')
                else:
                    self.issue_list=list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype,epoch,self.issue_list,\
                        str(time.time()-self.start_time),'Found Issue Stop Training! Starting the repair procedure.'))
                    self.log_file.flush()
                    self.log_file.close()
                    self.model.stop_training = True
            else:
                if self.issue_list==[]:
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype,epoch,self.issue_list,\
                        str(time.time()-self.start_time),'No Issue now'))
                    self.log_file.flush()
                elif self.issue_list==['need_train']:
                    self.issue_list=list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype,epoch,self.issue_list,\
                        str(time.time()-self.start_time),'NO training problems now. You need to train this new model more times.'))
                    self.log_file.flush()
                    self.log_file.write('-------------------------------\n')
                    self.log_file.flush()
                    #print('NO training problems now. You need to train this new model more iterations.')
                else:
                    self.issue_list=list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype,epoch,self.issue_list,\
                        str(time.time()-self.start_time),'Found Issue Stop Training! Starting the repair procedure.'))
                    self.log_file.flush()
                    self.log_file.write('-------------------------------\n')
                    self.log_file.flush()
                    self.model.stop_training = True


    def on_train_end(self,logs=None):
        if self.retrain==True and self.issue_list==[]:
            self.log_file.write('------------Solved!-----------\n')
            self.log_file.flush()

        solution_dir=os.path.join(self.save_dir,'solution')
        if self.retrain==True:
            solution_dir=self.pkl_dir
        if not os.path.exists(solution_dir):
            os.makedirs(solution_dir)
        issue_path=os.path.join(solution_dir,'issue_history.pkl')
        tmpset={'issue_list':self.issue_list,'history':self.history}
        with open(issue_path, 'wb') as f:
            pickle.dump(tmpset, f)
        self.log_file.close()
        print('Finished Training')


def save_model(model,path):
    """[summary]

    Args:
        model ([model]): [a model you want to save]
        path ([str]): [the path you want to save the model]
    """
    try:
        model.save(path,model)
        model.summary()
        logger.info('Saved model!')
    except:
        logger.error(sys.exc_info())

def has_NaN(output):
    output=np.array(output)
    result=(np.isnan(output).any() or np.isinf(output).any())
    return result

def generate_fig(array_dic,path,method=2):

    """
    :params array_dic: a dictionary contains multi-arrays, was used to be the data of the figure 
    :params path: a string, the path you want to save the fig
    :params method: int method. 1 means only one figure, 2 means 121,122 subplot
    """
    if method==1:
        plt.figure(figsize=(9, 6))
        plt.subplot(121)
        a=[]
        for key,value in array_dic.items():
            a.append(value)
        assert len(a)==2
        plt.plot(a[0], label='train')
        plt.plot(a[1], label='test')
        plt.xlabel('epochs', fontsize=18)
        plt.ylabel('indicator', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
    if method==2:
        plt.figure(figsize=(16, 6))
        plt.subplot(121)
        plt.plot(array_dic['acc'], label='train')
        plt.plot(array_dic['val_acc'], label='test')
        plt.xlabel('epochs', fontsize=18)
        plt.ylabel('accuracy', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.subplot(122)
        plt.plot(array_dic['loss'], label='train')
        plt.plot(array_dic['val_loss'], label='test')
        plt.xlabel('epochs', fontsize=18)
        plt.ylabel('loss', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

    plt.savefig(path,dpi=300)

def read_csv(csv_path,epoch):
    csvFile = open(csv_path, 'r')
    reader = csv.reader(csvFile)
    result = []
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        item = [float(x) for x in item]
        result.append(item)
    csvFile.close()
    x_axis = []
    with open(csv_path, 'r') as f:
        tmp=len(f.readlines())-1
        if tmp <= epoch:
            epoch=tmp
    for i in range(epoch):
        x_axis.append(i)
    while ([] in result):
        result.remove([])
    result = np.array(result)
    tmp_dic={}
    tmp_dic['acc']=result[:, 3]
    tmp_dic['val_acc']=result[:,1]
    tmp_dic['loss']=result[:,2]
    tmp_dic['val_loss']=result[:,0]
    return tmp_dic

def check_point_model(model_dir,model_name,config,history):
    model=load_model(model_name)
    test_acc=max(history.history['val_accuracy'])
    model_path=os.path.join(model_dir,'best_model_{}.h5'.format(test_acc))
    model.save(model_path)
    os.remove(model_name)

def pack_train_config(opt,loss,dataset,epoch,batch_size,callbacks):
    config={}
    config['opt']=opt
    config['loss']=loss
    config['dataset']=dataset
    config['epoch']=epoch
    config['batch_size']=batch_size
    config['callbacks']=callbacks
    return config

def model_train(model,
                train_config_set,
                optimizer,
                loss,
                dataset,
                iters,
                batch_size,
                log_dir,
                callbacks,
                root_path,
                new_issue_dir,
                verb=0,
                determine_threshold=1,
                save_dir='./tool_log',
                checktype='epoch_3',
                autorepair=True,
                modification_sufferance=3,#0-3 for model
                memory_limit=False,
                satisfied_acc=0.7,
                strategy='balance',
                params={}
                ):
    """[summary]
    Args:
        model ([model loaded by keras or str]): [a model you want to train or a model path(string)]
        optimizer ([str]): [the optimizer you want to use]
        loss ([str]): [usually 'categorical_crossentropy' or 'binary_crossentropy']
        dataset ([dic]): [a dictionary which contains 'x''y''x_val''y_val']
        iters ([int]): [max iterations in training]
        batch_size ([int]): [batch_size in training]
        log_path ([str]): [the path you want to save the training log]
        callbacks ([list]): [a list of the callbacks you want to use in the training. e.g., tensorboard , reducelr, earlystop]
        verb (int, optional): [model.fit, verbose]. Defaults to 0.
        save_dir (str, optional): [the dir you want to save all result(include the training report, trained model with each solution)].\
            Defaults to './tmp'.
        checktype (str, optional): ['a_b', a can be chosen from ['epoch', 'batch'], b is number, it means the monitor will check \
            the gradient and loss every 'b' 'a'.]. Defaults to 'epoch_3'.
        autorepair (bool, optional): [whether the user want our tools to auto repair or not, if not our tools will return the problems \
            and corresponding solutions, if yes, will return trained model and description and logs ]. Defaults to True.
        modification_sufferance (int, optional): [description]. Defaults to 3.
        satisfied_acc(float,optional):the satisfied accuracy in training, it will be used to dertermine if it has converged.
        strategy (str, optional): [chosen from ['balance','efficient','structure',it will determine the solution order when solving the problem ]]. Defaults to 'balance'.

    Returns:
        [type]: [if autorepair is True, return a trained model and the log/description file path.\
            if autorepair is False, only return the problems and the corresponding solution description]
    """
    save_dir = os.path.abspath(save_dir)
    log_dir=os.path.abspath(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if isinstance(model, str):
        model_path = model
        model = load_model(model_path)
    #K.clear_session()
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    callbacks = [n for n in callbacks if (
        n.__class__ != LossHistory and n.__class__ != ModelCheckpoint)]

    if 'estop' in callbacks:
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=3, 
            verbose=0, mode='auto',baseline=None, restore_best_weights=False))
        callbacks.remove('estop')
    if 'ReduceLR' in callbacks:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
            patience=5, min_lr=0.001))
        callbacks.remove('ReduceLR')
    
    
    checkpoint_name = "train_best.h5"
    checkpoint_dir = os.path.join(save_dir, 'checkpoint_model')
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    callbacks.append(ModelCheckpoint(
        checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max'))
    callbacks.append(LossHistory(training_data=[dataset['x'], dataset['y']], model=model,determine_threshold=determine_threshold,
                                 batch_size=batch_size, save_dir=save_dir, total_epoch=iters, satisfied_acc=satisfied_acc, checktype=checktype,params=params))  # issue in lstm network

    callbacks_new = list(set(callbacks))
    history = model.fit(dataset['x'], dataset['y'], batch_size=batch_size, validation_data=(
        dataset['x_val'], dataset['y_val']), epochs=iters, verbose=verb, callbacks=callbacks_new)
    check_point_model(checkpoint_dir, checkpoint_path, dataset,history)

    result = history.history
    time_callback = TimeHistory()
    log_path = os.path.join(log_dir, 'log.csv')
    if 'val_loss' in result.keys():
        time_callback.write_to_csv(result, log_path, iters)

    solution_dir = os.path.join(save_dir, 'solution')
    issue_path = os.path.join(solution_dir, 'issue_history.pkl')
    with open(issue_path, 'rb') as f:  #input,bug type,params
        output = pickle.load(f)
    issues = output['issue_list']
    if issues!=[]:
        if autorepair == True:
            #auto repair
            train_config = pack_train_config(optimizer, loss, dataset, iters,
                                                batch_size, callbacks)
            start_time=time.time()
            rm = md.Repair_Module(
                model=model,
                training_config=train_config,
                issue_list=issues,
                sufferance=modification_sufferance,
                memory=memory_limit,
                satisfied_acc=satisfied_acc,
                checktype=checktype,
                determine_threshold=determine_threshold,
                config_set=train_config_set,
                root_path=root_path
            )  #train_config need to be packed and issue need to be read.
            result,model,trained_path,test_acc,history,issue_list,now_issue = rm.solve(solution_dir,new_issue_dir=new_issue_dir)

            tmpset={}
            tmpset['time']=time.time()-start_time
            tmpset['test_acc']=test_acc
            tmpset['model_path']=trained_path
            tmpset['history']=history
            tmpset['initial_issue']=issue_list
            tmpset['now_issue']=now_issue
            tmppath=os.path.join(save_dir,'repair_result_total.pkl')
            with open(tmppath, 'wb') as f:
                pickle.dump(tmpset, f)
        else:
            print('You can find the description of the solution candidates in {}'.format('./path'))
    return result,model, trained_path


def model_retrain(model,
                config,
                satisfied_acc,
                save_dir,
                retrain_dir,
                verb=1,
                solution=None,
                determine_threshold=5,
                checktype='epoch_3'
                ):
    retrain_dir=os.path.abspath(retrain_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if isinstance(model,str):
        model_path=model
        model=load_model(model_path)
    model.compile(loss=config['loss'], optimizer=config['opt'], metrics=['accuracy'])
    config['callbacks'] = [n for n in config['callbacks'] if (n.__class__!=LossHistory and n.__class__!=ModelCheckpoint)]
    config['callbacks'].append(
        LossHistory(training_data=[config['dataset']['x'], config['dataset']['y']],
                    model=model,
                    batch_size=config['batch_size'],
                    save_dir=retrain_dir,
                    pkl_dir=save_dir,
                    total_epoch=config['epoch'],
                    determine_threshold=determine_threshold,
                    checktype=checktype,
                    satisfied_acc=satisfied_acc,
                    retrain=True,
                    solution=solution,params={}))  
    checkpoint_name="train_best.h5"
    checkpoint_dir=os.path.join(save_dir,'checkpoint_model')
    checkpoint_path=os.path.join(checkpoint_dir,checkpoint_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    config['callbacks'].append(ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max'))
    callbacks_new=list(set(config['callbacks']))
    history = model.fit(config['dataset']['x'], config['dataset']['y'],batch_size=config['batch_size'], validation_data=(config['dataset']['x_val'], config['dataset']['y_val']),\
        epochs=config['epoch'], verbose=verb,callbacks=callbacks_new)
    check_point_model(checkpoint_dir,checkpoint_path,config,history)
    issue_path = os.path.join(save_dir, 'issue_history.pkl')   
    with open(issue_path, 'rb') as f:  
        output = pickle.load(f)
    new_issues = output['issue_list']
    if 'need_train' in new_issues:
        new_issues=[]
    test_acc=history.history['val_accuracy'][-1]
    return model,new_issues,test_acc,history.history
