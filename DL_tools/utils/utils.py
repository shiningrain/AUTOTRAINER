import os
import sys
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

#when status>5 consider as dyingrelu
#gradient vanish need to update, vanish still can converge but it converges too slow
#loss converge need to be update

class LossHistory(keras.callbacks.Callback):

    def __init__(self,training_data,model,batch_size,total_epoch,save_dir,expect_acc,\
        checktype='epoch_5',satisfied_acc=0.7,satisfied_count=3): #only support epoch method now
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
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.expect_acc=expect_acc

        self.history={}
        self.history['loss']=[]
        self.history['acc']=[]
        self.history['val_loss']=[]
        self.history['val_acc']=[]

        self.totolratio=[]
        self.maxratio=[]
        self.gradient_rate=[]

        self.start_time=time.time()
        log_name='{}_{}.log'.format('monitor',str(time.time()))
        log_name=os.path.join(self.save_dir,log_name)
        self.log_file=open(log_name,'a+')
        self.log_file.write('{},{},{},{},{}.log\n'.format('checktype','current_epoch','issue_list','time_usage','Describe'))

        self.Monitor=mn.IssueMonitor(total_epoch,satisfied_acc)

    def on_train_begin(self,logs=None):
        weights=self.model.trainable_weights# get trainable weights
        grads = self.model.optimizer.get_gradients(self.model.total_loss, weights)
        symb_inputs = [self.model._feed_inputs , self.model._feed_targets , self.model._feed_sample_weights,K.learning_phase()]#input,corresponding label,weight of each sample(all of them are 1),learning rate(we set it to 0)
        self.f = K.function(symb_inputs, grads)
        print('start')

    def on_epoch_begin(self,epoch,logs={}):#加入对loss和acc的观测方法
        #global Dying_ReLU_status,Gradient_explode_status,Gradient_vanish_status
        if (epoch)%self.checkgap==0:
            trainingExample = self.trainX[0:self.batch_size,...]
            trainingY=self.trainy[0:self.batch_size]
            x, y, sample_weight = self.model._standardize_user_data(trainingExample, trainingY)
            #output_grad = f(x + y + sample_weight)
            self.evaluated_gradients = self.f([x , y , sample_weight,0])
            #For every epoch, we only collect the kernel gradient information: 1) The ratio of 0 in each layer.2)the ratio of 0 in all layers. \
            # 3) the average gradient of each layer
            # gradient_exp,gradient_van=mn.gradient_issue(evaluated_gradients,threshold_low,threshold_high)
            # if gradient_exp: Gradient_explode_status+=1
            # if gradient_van: Gradient_vanish_status+=1
            # avg_kernel,avg_bias=average_gradient(evaluated_gradients)
            # gradient_arr=evaluated_gradients[0]
            self.issue_list=self.Monitor.determine(self.model,self.history,self.evaluated_gradients)

            self.issue_list=md.filtered_issue(self.issue_list)

            if self.issue_list==[]:
                self.log_file.write('{},{},{},{},{}\n'.format(self.checktype,epoch,self.issue_list,\
                    str(time.time()-self.start_time),'No Issue now'))
                self.log_file.flush()
            else:
                self.issue_list=list(set(self.issue_list))
                self.log_file.write('{},{},{},{},{}\n'.format(self.checktype,epoch,self.issue_list,\
                    str(time.time()-self.start_time),'Found Issue Stop Training! Starting the repair procedure.'))
                self.log_file.flush()
                self.log_file.close()
                solution_dir=os.path.join(self.save_dir,'solution')
                if not os.path.exists(solution_dir):
                    os.makedirs(solution_dir)
                issue_path=os.path.join(solution_dir,'issue.pkl')
                tmpset={'issue_list':self.issue_list}
                with open(issue_path, 'wb') as f:
                    pickle.dump(tmpset, f)
                #repair the model now
            #self.gradient_rate.append(avg_kernel[0]/avg_kernel[-1])
            #print(1)
        # if max_zero>=0.6:
        #     Dying_ReLU_status+=1
        #print(epoch)



    def on_batch_begin(self,batch,logs={}):#加入对loss和acc的观测方法
        #global Dying_ReLU_status,Gradient_explode_status,Gradient_vanish_status
        if (batch-1)%self.checkgap==0:
            trainingExample = self.trainX[0:self.batch_size,...]
            trainingY=self.trainy[0:self.batch_size]
            x, y, sample_weight = self.model._standardize_user_data(trainingExample, trainingY)
            #output_grad = f(x + y + sample_weight)
            evaluated_gradients = self.f([x , y , sample_weight,0])
            [self.avg_kernel,self.avg_bias,self.fb_gradient_rate],\
                [self.total_ratio,self.kernel_ratio,self.bias_ratio,self.max_zero]\
                    =mn.gradient_message_summary(self.evaluated_gradients)

    # def on_batch_end(self,batch,logs={}):
    #     outputTensor =  self.model.output
    #     listOfVariableTensors =  self.model.trainable_weights
    #     print(logs.get('loss'))
    #     print(logs.get('val_loss'))

    def on_epoch_end(self,epoch,logs={}):
        # in each epoch we save the history message.
        outputTensor =  self.model.output
        listOfVariableTensors =  self.model.trainable_weights
        self.history['loss'].append(logs.get('loss'))
        self.history['acc'].append(logs.get('accuracy'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_acc'].append(logs.get('val_accuracy'))
        # self.loss.append(logs.get('loss'))
        # self.accuracy.append(logs.get('accuracy'))
        # self.val_loss.append(logs.get('val_loss'))
        # self.val_accuray.append(logs.get('val_accuracy'))
        # 在结尾做一个检查，检查问题，首先检查gradient信息，有问题了再检查loss信息，都有问题就停止训练进行修复。

        #if epoch==:issue=True
        #else: issue=False
        #--------------------------checkissue here-----------------------IssueMonitor
        if len(self.issue_list)>0:
            self.model.stop_training = True

    def on_train_end(self,logs=None):
        # try:
        #     with open("./vanish_no_1.pkl", 'rb') as f:#input,bug type,params
        #         tmpset = pickle.load(f)
        # except:
        #     tmpset={}
        #     tmpset['gradient_rate']=[]
        # #     tmpset['ave']=[]
        # #     tmpset['max']=[]
        # # tmpset['ave'].append(np.average(self.totolratio))
        # # tmpset['max'].append(np.average(self.maxratio))
        # tmpset['gradient_rate'].append(np.average(self.gradient_rate))
        # with open("./vanish_no_1.pkl", 'wb') as f:
        #     pickle.dump(tmpset, f)
        # print('start')
        self.log_file.close()
        print('Finished Training')

# def get_layer_output_grad(model, inputs, outputs, layer=-1):
#     """ Gets gradient a layer output for given inputs and outputs"""
#     grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
#     symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
#     f = K.function(symb_inputs, grads)
#     x, y, sample_weight = model._standardize_user_data(inputs, outputs)
#     output_grad = f(x + y + sample_weight)
#     return output_grad


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
                optimizer,
                loss,
                dataset,
                iters,
                batch_size,
                log_path,
                callbacks,
                verb=0,
                save_dir='./tmp',
                checktype='epoch_3',
                autorepair=True,
                modification_sufferance=3,#0-3 for model
                memory_limit=False,
                satisfied_acc=0.7,
                strategy='balance'
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
    save_dir=os.path.abspath(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if isinstance(model,str):
        model_path=model
        model=load_model(model_path)
    #K.clear_session()
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    lh=0
    mc=0
    for cb in callbacks:
        if cb.__class__==LossHistory:lh=1
        if cb.__class__==ModelCheckpoint:mc=1
    if lh==0:
        callbacks.append(LossHistory(training_data=[dataset['x'],dataset['y']],model=model,\
            batch_size=batch_size,save_dir=save_dir,total_epoch=iters,expect_acc=satisfied_acc))# issue in lstm network
    if mc==0:
        checkpoint_name='train_best.h5'
        checkpoint_path=os.path.join(save_dir,checkpoint_name)
        callbacks.append(ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min'))
    callbacks_new=list(set(callbacks))
    history = model.fit(dataset['x'], dataset['y'],batch_size=batch_size, validation_data=(dataset['x_val'], dataset['y_val']), epochs=iters, verbose=verb,callbacks=callbacks_new)#,callbacks=[reducelr,tensorboard_callback])
    solution_dir=os.path.join(save_dir,'solution')
    if os.path.exists(solution_dir):
        if autorepair == True:
            #auto repair
            train_config = pack_train_config(optimizer, loss, dataset, iters,
                                             batch_size, callbacks)
            issue_path = os.path.join(solution_dir, 'issue.pkl')
            with open(issue_path, 'rb') as f:  #input,bug type,params
                output = pickle.load(f)
            issues = output['issue_list']
            rm = md.Repair_Module(
                model=model,
                training_config=train_config,
                #pure_config=train_config,
                issue_list=issues,
                sufferance=modification_sufferance,
                memory=memory_limit,
                satisfied_acc=satisfied_acc
            )  #train_config need to be packed and issue need to be read.
            result,model,trained_path = rm.solve(solution_dir)
        else: 
            print('see the ./path')# return a describe in solution
    else:
        print('Model has NO issue.')# no issue
        result = history.history
        time_callback = TimeHistory()
        time_callback.write_to_csv(result,log_path,iters)
        # evaluate the model
        _, train_acc = model.evaluate(dataset['x'], dataset['y'], verbose=0)
        _, test_acc = model.evaluate(dataset['x_val'], dataset['y_val'], verbose=0)
        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
        tmp = log_path.split('/')[-1]
        model_name = 'trained_' + str(round(test_acc, 3)) + '.h5'
        trained_path = log_path.replace(tmp, model_name)
        save_model(model, trained_path)
        result='no'
    # #K.clear_session()
    # wts = model.get_weights()
    return result,model, trained_path#history, train_acc  #加上保存trained model的功能


def model_retrain(model,
                config,
                satisfied_acc,
                verb=0,
                retrain_dir='./tmp',):
    retrain_dir=os.path.abspath(retrain_dir)
    if isinstance(model,str):
        model_path=model
        model=load_model(model_path)
    #K.clear_session()
    '''
    config['opt']=opt
    config['loss']=loss
    config['dataset']=dataset
    config['epoch']=epoch
    config['batch_size']=batch_size
    config['callbacks']=callbacks
    '''
    model.compile(loss=config['loss'], optimizer=config['opt'], metrics=['accuracy'])
    # tmp_length=len(config['callbacks'])
    # # for cb in range(tmp_length):
    # #     if config['callbacks'][cb].__class__==LossHistory :
    # #         lh=cb      
    # #     if cb.__class__==ModelCheckpoint:
    # #         mc=cb
    # #         config['callbacks'].remove(cb)
    config['callbacks'] = [n for n in config['callbacks'] if (n.__class__!=LossHistory and n.__class__!=ModelCheckpoint)]
    config['callbacks'].append(
        LossHistory(training_data=[config['dataset']['x'], config['dataset']['y']],
                    model=model,
                    batch_size=config['batch_size'],
                    save_dir=retrain_dir,
                    total_epoch=config['epoch'],
                    expect_acc=satisfied_acc))  # issue in lstm network
    checkpoint_name='retrain_best.h5'
    checkpoint_path=os.path.join(retrain_dir,checkpoint_name)
    config['callbacks'].append(ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min'))
    callbacks_new=list(set(config['callbacks']))
    history = model.fit(config['dataset']['x'], config['dataset']['y'],batch_size=config['batch_size'], validation_data=(config['dataset']['x_val'], config['dataset']['y_val']),\
        epochs=config['epoch'], verbose=verb,callbacks=callbacks_new)#,callbacks=[reducelr,tensorboard_callback])
    solution_dir=os.path.join(retrain_dir,'solution')
    issue_path = os.path.join(solution_dir, 'issue.pkl')
    if os.path.exists(solution_dir):
        with open(issue_path, 'rb') as f:  #input,bug type,params
            output = pickle.load(f)
        new_issues = output['issue_list']
    else: new_issues=[]
    _, test_acc = model.evaluate(config['dataset']['x_val'], config['dataset']['y_val'], verbose=0)
    return model,new_issues,test_acc

#特征0/1判断，问题根据特征触发比例来计。

#1.回馈中得以体现——对问题解决效果的综合评价，而非绝对的解决与否。
#2.