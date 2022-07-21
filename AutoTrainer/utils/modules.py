import os
import sys
sys.path.append('.')
import utils as utils
import copy
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import datetime
import repair as rp
from TimeCounter import TimeHistory
from tensorflow.keras.models import load_model,Sequential
import tensorflow.keras.backend as K
import tensorflow as tf
from logger import Logger
logger = Logger()
import time
import uuid
from collections import Counter
import copy
import csv
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu,sigmoid,elu,linear,selu
from tensorflow.keras.layers import BatchNormalization,GaussianNoise
from tensorflow.keras.layers import Activation,Add,Dense,Softmax
from tensorflow.keras.layers import Lambda
from tensorflow.keras.initializers import he_uniform,glorot_uniform,zeros
from tensorflow.keras.optimizers import SGD, Adam, Adamax
from tensorflow.keras.callbacks import ReduceLROnPlateau

solution_evaluation = {
        'gradient' : {'modify':0, 'memory':False},
        'relu': {'modify':1, 'memory':False},
        'bn': {'modify':2, 'memory':True},
        'initial': {'modify':1, 'memory':False},
        'selu': {'modify':1 , 'memory':False},
        'tanh': {'modify':1 , 'memory':False},
        'leaky': {'modify':2 , 'memory':False},
        'adam' : {'modify':0 , 'memory':False},
        'lr': {'modify':0 , 'memory':False},
        'ReduceLR': {'modify':0 , 'memory':False},
        'momentum' : {'modify':0 , 'memory':False},
        'batch': {'modify':0 , 'memory':False},
        'GN': {'modify':2 , 'memory':False},
        'optimizer': {'modify':0 , 'memory':False},
        'regular':{'modify':1 , 'memory':False},#how
        'dropout':{'modify':2 , 'memory':False},#how
        'estop':{'modify':0, 'memory':False},
        'activation':{'modify':0, 'memory':False},
        'loss':{'modify':0, 'memory':False},
        'data':{'modify':0, 'memory':False}
    }
problem_evaluation = {# please use the priority order here
    'activation_issue':3,
    'loss_issue':3,
    'abnormal_data':3,
    'vanish': 2,
    'explode': 2,
    'relu': 2,
    'not_converge': 1,
    'unstable': 1,
    'overfit': 0,
    'need_train':0
}

def csv_determine(issue_name,path):
    file_name=issue_name+'.csv'
    file_path=os.path.join(path,file_name)
    return file_path

def filtered_issue(issue_list):
    if len(issue_list)>1:
        new_issue_list=[]
        for i in range(len(issue_list)):
            if new_issue_list==[] or problem_evaluation[issue_list[i]]>problem_evaluation[new_issue_list[0]]:
                new_issue_list=[issue_list[i]]
        return new_issue_list
    return issue_list

def comband_solutions(solution_dic):
    solution=[]
    stop=False
    i=0
    while(stop==False):
        solution_start=len(solution)
        for key in solution_dic:
            if i <= (len(solution_dic[key])-1):
                solution.append(solution_dic[key][i])
        if solution_start==len(solution): stop=True
    return solution


def filtered(strategy_list,sufferance,memory):
    for strat in range(len(strategy_list)):
        for i in range(len(strategy_list[strat])):
            sol=strategy_list[strat][i].split('_')[0]
            if solution_evaluation[sol]['modify']>sufferance or (memory==True and solution_evaluation[sol]['memory']==True ):
                if len(strategy_list[strat])>1:
                    strategy_list[strat].remove(strategy_list[strat][i])
                else: print('-----WARNING, TOO STRICT FILTER! NOW KEEPED ONLY ONE SOLUTION!-----')
    return strategy_list

def merge_history(history,new_history):
    if history=={}:
        history=new_history.copy()
        history['train_node']=[len(new_history['loss'])]
        return history
    for i in history.keys():
        if i in new_history.keys():
            for j in range(len(new_history[i])):
                history[i].append(new_history[i][j])
    history['train_node'].append(len(new_history['loss']))
    return history

def read_strategy(string):
    solution=string.split('_')[0]
    times= string.split('_')[-1]
    #if string.split('_')[-1]=='': times=1
    return solution,int(times)


def get_new_dir(new_issue_dir,case_name,issue_type,tmp_add):
    case_name=case_name.split('/')[-1]
    new_case_name=case_name+'-'+tmp_add
    new_issue_type_dir=os.path.join(new_issue_dir,issue_type)
    new_case_dir=os.path.join(new_issue_type_dir,new_case_name)
    if not os.path.exists(new_case_dir):
        os.makedirs(new_case_dir)
    return new_case_dir

def notify_result(num, model,config,issue,j,config_set):
    numbers = {
        'gradient' : rp.op_gradient,
        'relu': rp.op_relu,
        'bn': rp.op_bn,
        'initial': rp.op_initial,
        'selu': rp.op_selu,
        'leaky': rp.op_leaky,
        'adam' : rp.op_adam,
        'lr': rp.op_lr,
        'ReduceLR': rp.op_ReduceLR,
        'momentum' : rp.op_momentum,
        'batch': rp.op_batch,
        'GN': rp.op_GN,
        'optimizer': rp.op_optimizer,
        'regular':rp.op_regular,
        'dropout':rp.op_dropout,
        'estop':rp.op_EarlyStop,
        'tanh':rp.op_tanh,
        'loss':rp.op_loss,
        'activation':rp.op_activation,
        'data':rp.op_preprocess
    }

    method = numbers.get(num, rp.repair_default)
    if method:
        return method(model,config,issue,j,config_set)

class Repair_Module:
    def __init__(self, config_set, model, training_config, issue_list, sufferance, memory, satisfied_acc, root_path,method='balance', checktype='epoch_3', determine_threshold=5):
        """#pure_config,
        method:['efficiency','structure','balance'], efficient will try the most efficiently solution and the structure will
            first consider to keep the model structure/training configuration.balance is the compromise solution.
        """
        #self.pure_config=pure_config
        self.initial_time = time.time()

        self.satisfied_acc = satisfied_acc
        self.model = model
        self.issue_list = issue_list
        self.root_path=root_path
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        self.initial_issue = copy.deepcopy(issue_list)
        self.train_config = training_config
        self.config_bk = training_config.copy()
        self.best_potential = []
        self.checktype = checktype
        self.determine_threshold = determine_threshold
        #if len(issue_list)>=2: self.multi=True
        #'_x' in strategy means this solution will be tried at most x times.
        strategy_list = rp.repair_strategy(method)
        self.gradient_vanish_strategy, self.gradient_explode_strategy, self.dying_relu_strategy,\
            self.unstable_strategy, self.not_converge_strategy, self.over_fitting_strategy,\
                self.activation_strategy,self.loss_strategy,self.data_strategy = filtered(
                strategy_list, sufferance, memory)
        self.config_set_bk = config_set
        


    def solve(self,tmp_dir,new_issue_dir): # need multi solve + retry
        #solve: new_model,train_history,save_dir,train_result,config,model_path,log_name
        #no: [new_model,train_history,config,new_issue_list,train_result,log_name,model_path]
        #potential:[new_model,retrain_history,config,new_issue_list,train_result,log_name,solution_dir]
        if not os.path.exists(new_issue_dir):
            os.makedirs(new_issue_dir)
        solved_issue=[]
        history={}
        # tmp_add=''
        for i in range(10):
            #case_name=tmp_dir.replace('/monitor_tool_log/solution','')
            case_name=tmp_dir.replace(('/'+tmp_dir.split('/')[-1]),'')
            case_name=case_name.replace(('/'+case_name.split('/')[-1]),'')
            if len(self.issue_list)==1:# only one issue to be solved:
                self.csv_file = csv_determine(self.issue_list[0],self.root_path)
                result, result_list,new_config_set= self.issue_repair(
                    self.model, self.train_config, self.config_bk, tmp_dir, \
                    self.issue_list,\
                    csv_path=self.csv_file,case_name=case_name)#, tmp_add=tmp_add
                train_history=merge_history(history,result_list[1])
                solved_issue.append(self.issue_list[0])
                if result == 'solved':

                    tmpset={}
                    tmpset['time']=time.time()-self.initial_time
                    tmpset['test_acc']=result_list[3]
                    tmpset['model_path']=result_list[-2]
                    tmpset['history']=result_list[1]
                    tmpset['initial_issue']=solved_issue
                    tmpset['now_issue']=[]
                    tmpset['result']=result
                    save_dir=tmp_dir.replace('/solution','')
                    tmppath=os.path.join(save_dir,'repair_result_single.pkl')# save in each case monitor_tool_log
                    with open(tmppath, 'wb') as f:
                        pickle.dump(tmpset, f)


                    print('Your model has been trained and the training problems {} have been repaired,you can find the repair log in {}, and the trained model is saved in {}.'.format(self.initial_issue, result_list[-1],
                                                                                                                                                                                        result_list[-2]))
                    return result,result_list[0],result_list[-2],result_list[3],train_history,solved_issue,[]
                    #   result,    model,   trained_path,   test_acc,   history,issue_list,now_issue 
                elif result == 'no' and i==0:
                    tmpset={}
                    tmpset['time']=time.time()-self.initial_time
                    tmpset['test_acc']=result_list[-3]
                    tmpset['model_path']=result_list[-1]
                    tmpset['history']=result_list[1]
                    tmpset['initial_issue']=solved_issue
                    tmpset['now_issue']=self.issue_list
                    tmpset['result']=result
                    save_dir=tmp_dir.replace('/solution','')
                    tmppath=os.path.join(save_dir,'repair_result_single.pkl')# save in each case monitor_tool_log
                    with open(tmppath, 'wb') as f:
                        pickle.dump(tmpset, f)
                    print(
                        'Your model still has training problems {} are still exist, you can try other solutions, see the file in {}, repair log can be found in {}'
                        .format(result_list[3], '/path',result_list[-2]))
                    return result,result_list[0],result_list[-1],result_list[-2],train_history,solved_issue,result_list[3]
                else:#elif result == 'potential':
                    #[new_model,config,new_issue_list,train_result,log_name]
                    if result=='no':

                        tmpset={}
                        tmpset['time']=time.time()-self.initial_time
                        tmpset['test_acc']=result_list[-3]
                        tmpset['model_path']=result_list[-1]
                        tmpset['history']=result_list[1]
                        tmpset['initial_issue']=solved_issue
                        tmpset['now_issue']=self.issue_list
                        tmpset['result']=result
                        save_dir=tmp_dir.replace('/solution','')
                        tmppath=os.path.join(save_dir,'repair_result_single.pkl')# save in each case monitor_tool_log
                        with open(tmppath, 'wb') as f:
                            pickle.dump(tmpset, f)

                        model_name='improved_unsolved_model_'+str(self.best_potential[-3])+'.h5'
                        model_path=os.path.join(tmp_dir,model_name)
                        self.best_potential[0].save(model_path)
                        print('Model has been improved but still has problem. The initial training problems in the source model is {}.\
                            The current problem is {}. You can find the repair log in {}, and the best improved model is saved \
                            in {}.'.format(self.initial_issue,self.best_potential[3],self.best_potential[-2],model_path))
                        return result,self.best_potential[0],model_path,self.best_potential[-3],train_history,solved_issue,self.best_potential[3]
                    
                    
                    potential_list=result_list
                    self.issue_list=potential_list[3]
                    self.model=potential_list[0]
                    self.train_config=potential_list[2]
                    model_name='improved_model_'+str(potential_list[-3])+'.h5'
                    model_path=os.path.join(tmp_dir,model_name)
                    # if os.path.exists(model_path):
                    #     os.remove(model_path)
                    try:
                        self.model.save(model_path)
                    except:
                        #TODO: solve this problem
                        print('Tensorflow Save Model Failed!! Save Without Optimizer Now!!! More Details ref to https://github.com/tensorflow/tensorflow/issues/27688')
                        self.model.save(model_path, include_optimizer=False)
                        # optimizer_config=
                        optimizer_path=model_path.replace('.h5','_optimizer.pkl')
                        optimizer_config={}
                        optimizer_config['optimizer']=self.model.optimizer.get_config()
                        optimizer_config['loss']=self.model.loss
                        with open(optimizer_path, 'wb') as f:
                            pickle.dump(optimizer_config, f)
                    
                    tmpset={}
                    tmpset['time']=time.time()-self.initial_time
                    tmpset['test_acc']=potential_list[-3]
                    tmpset['model_path']=model_path
                    tmpset['history']=potential_list[1]
                    tmpset['initial_issue']=solved_issue
                    tmpset['now_issue']=self.issue_list
                    tmpset['result']=result
                    save_dir=tmp_dir.replace('/solution','')
                    tmppath=os.path.join(save_dir,'repair_result_single.pkl')# save in each case monitor_tool_log
                    with open(tmppath, 'wb') as f:
                        pickle.dump(tmpset, f)
                        
                    if self.issue_list[0] not in solved_issue:
                        new_dir=get_new_dir(new_issue_dir,case_name,potential_list[3][0],potential_list[-1])
                        
                        #save new model
                        model_path=os.path.join(new_dir,'new_model.h5')
                        config_path=os.path.join(new_dir,'new_config.pkl')
                        try:
                            self.model.save(model_path)
                        except:
                            print('Tensorflow Save Model Failed!! Save Without Optimizer Now!!! More Details ref to https://github.com/tensorflow/tensorflow/issues/27688')
                            self.model.save(model_path, include_optimizer=False)

                        with open(config_path, 'wb') as f:
                            pickle.dump(new_config_set, f)
                        new_log_dir=os.path.join(new_dir,'monitor_train_log')
                        if not os.path.exists(new_log_dir):
                            os.makedirs(new_log_dir)
                        new_save_dir=os.path.join(new_dir,'monitor_tool_log')
                        if not os.path.exists(new_save_dir):
                            os.makedirs(new_save_dir)
                        common_log_path=os.path.join(new_log_dir,'common_log_history.pkl')

                        common_log_history={}
                        common_log_history['history']=result_list[1]
                        with open(common_log_path, 'wb') as f:
                            pickle.dump(common_log_history, f)
                        new_tmp_dir=os.path.join(new_save_dir,'solution')
                        if not os.path.exists(new_tmp_dir):
                            os.makedirs(new_tmp_dir)
                        issue_history_path=os.path.join(new_tmp_dir,'issue_history_before_repair.pkl')
                        issue_history={}
                        issue_history['issue']=self.issue_list[0]
                        issue_history['history']=result_list[1]
                        with open(issue_history_path, 'wb') as f:
                            pickle.dump(issue_history, f)
                        
                        if self.best_potential==[] or self.best_potential[-3]<potential_list[-3]:
                            self.best_potential=potential_list.copy()
                        if 'sp' in self.train_config.keys():#(num, model,config,issue,j,config_set):
                            self.model,self.train_config,_1,_2,_3=notify_result(self.train_config['sp'],self.model,self.train_config,0,0,self.config_bk.copy())
                            del self.train_config['sp']

                        self.config_bk=self.train_config.copy()
                        self.config_set_bk=new_config_set.copy()
                        tmp_dir=new_tmp_dir
                        history=train_history
                        # tmp_add=tmp_add+potential_list[-1]+'-'
                        del potential_list
                    else:
                        model_name='improved_model_'+str(self.best_potential[-3])+'.h5'
                        model_path=os.path.join(tmp_dir,model_name)
                        self.best_potential[0].save(model_path)
                        print('Model has been improved but still has problem. The initial training problems in the source model is {}.\
                            The current problem is {}. You can find the repair log in {}, and the best improved model is saved \
                            in {}.'.format(self.initial_issue,self.best_potential[3],self.best_potential[-2],model_path))
                        return result,self.best_potential[0],model_path,self.best_potential[-3],train_history,solved_issue,self.best_potential[3]


    def issue_repair(self,seed_model,train_config,config_bk,tmp_dir,issue_list,csv_path,case_name,max_try=10):#need modify tmp_add=''
        """[summary]

        Args:
            seed_model ([type]): [description]
            train_config ([type]): [description]
            config_bk ([type]): [description]
            tmp_dir ([type]): [description]
            issue_list (bool): [description]
            tmp_add (str, optional): [description]. Defaults to ''.
            max_try (int, optional): [Max try solution, if not solve in this solution and has potential, then try to solve the potential]. Defaults to 2.

        Returns:
            [type]: [description]
        """
        #file_name,solution_list
        #if len(issue_list)==1:
        #seed_model=copy.deepcopy(model)
        issue_type=issue_list[0]
        file_name,solution_list=self.get_file_name(issue_type)
        log_name='{}_{}.log'.format(file_name,str(time.time()))
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        log_name=os.path.join(tmp_dir,log_name)
        log_file=open(log_name,'a+')
        log_file.write('solution,times,issue_list,train_result,describe\n')
        start_time=time.time()
        potential=[]
        log=[]
        length_solution=min(len(solution_list),max_try)
        try_count=0

        
        if not os.path.exists(self.csv_file):
            tmp_list=solution_list.copy()
            tmp_list.insert(0,'case_name')
            with open(self.csv_file,"a+") as csvfile: 
                writer = csv.writer(csvfile)
                writer.writerow(tmp_list)

        effect_list=[case_name]
        for i in range(length_solution):
            effect_list.append('×')
            tmp_sol,tmp_tim=read_strategy(solution_list[i])
            self.update_current_solution(issue_type,solution_list[i])
            for j in range(tmp_tim):
                # solutions
                _break=False
                try:
                    model=copy.deepcopy(seed_model)
                except:
                    model=seed_model
                train_config=config_bk.copy()
                config_set=copy.deepcopy(self.config_set_bk)
                tmp_model,config,modify_describe,_break,new_config_set=notify_result(tmp_sol,model,train_config,issue_type,j,config_set)
                if _break: break#  the solution has already been used in the source model.
                print('-------------Solution {} has been used, waiting for retrain.-----------'.format(tmp_sol))
                
                solution_dir=str(issue_type)+'_'+str(tmp_sol)+'_'+str(j)
                save_dir=os.path.join(tmp_dir,solution_dir)
                repair_time=time.time()-start_time
                
                # TODO:load previous acc as satisfy acc
                new_model,new_issue_list,train_result,retrain_history=utils.model_retrain(tmp_model,config,satisfied_acc=self.satisfied_acc,\
                    retrain_dir=tmp_dir,save_dir=save_dir,solution=tmp_sol,determine_threshold=self.determine_threshold,checktype=self.checktype)

                # new_issue_list=['overfit']
                # new_issue_list=[np.random.choice(['vanish','unstable','explode','relu'])]
                
                if tmp_sol=='estop' or tmp_sol=='ReduceLR':#special operation for this two solution will will change the config_bk
                    config_bk['callbacks'].remove(config_bk['callbacks'][-1])
                    config['sp']=tmp_sol
                
                log_file.write('{},{},{},{},{}\n'.format(tmp_sol,j,new_issue_list,train_result,modify_describe))
                log_file.flush()

                if new_issue_list==[]:
                    end_time=time.time()
                    effect_list[i+1]='√'
                    time_used=end_time-start_time
                    
                    model_name='solved_model_'+str(train_result)+'.h5'
                    new_config_name='sovled_config.pkl'
                    model_path=os.path.join(save_dir,model_name)
                    new_config_path=os.path.join(save_dir,new_config_name)
                    with open(new_config_path, 'wb') as f:
                        pickle.dump(new_config_set, f)
                    try:
                        new_model.save(model_path)
                    except:
                        print('Tensorflow Save Model Failed!! Save Without Optimizer Now!!! More Details ref to https://github.com/tensorflow/tensorflow/issues/27688')
                        new_model.save(model_path, include_optimizer=False)

                    print('------------------Solved! Time used {}!-----------------'.format(str(time_used)))
                    log_file.write('------------------Solved! Time used {}!-----------------'.format(str(time_used)))
                    with open(self.csv_file,"a+") as csvfile: 
                        writer = csv.writer(csvfile)
                        writer.writerow(effect_list)
                    log_file.close()
                    return 'solved', [new_model,retrain_history,save_dir,train_result,config,model_path,log_name],new_config_set
                
                elif issue_type not in new_issue_list :
                    effect_list[i+1]='○'
                    if(potential==[] or potential[-3]<train_result):
                        potential=[new_model,retrain_history,config,new_issue_list,train_result,log_name,solution_dir]
                        potential_config_set=copy.deepcopy(new_config_set)
                else:
                    effect_list[i+1]='×'
                if log==[] or log[-2]<=train_result:
                    log=[new_model,retrain_history,config,new_issue_list,train_result,log_name]
                    log_config_set=copy.deepcopy(new_config_set)
            

            if try_count>=max_try:# if tried solution is over maxtry, then go to solve the potential type.
                break

        if potential==[]: 
            end_time=time.time()
            time_used=end_time-start_time
            model_name='unsolved_model_'+str(log[-2])+'.h5'
            new_config_name='unsovled_config.pkl'
            model_path=os.path.join(save_dir,model_name)
            new_config_path=os.path.join(save_dir,new_config_name)
            with open(new_config_path, 'wb') as f:
                pickle.dump(new_config_set, f)
            log[0].save(model_path)
            log.append(model_path)
            print('------------------Unsolved..Time used {}!-----------------'.format(str(time_used)))
            log_file.write('------------------Unsolved..Time used {}!-----------------'.format(str(time_used)))
            log_file.close()
            with open(self.csv_file,"a+") as csvfile: 
                writer = csv.writer(csvfile)
                writer.writerow(effect_list)
            return 'no',log,log_config_set

        else:
            end_time=time.time()
            time_used=end_time-start_time
            print('------------------Not totally solved..Time used {}!-----------------'.format(str(time_used)))
            log_file.write('------------------Not totally solved..Time used {}!-----------------'.format(str(time_used)))
            log_file.close()
            with open(self.csv_file,"a+") as csvfile: 
                writer = csv.writer(csvfile)
                writer.writerow(effect_list)
            return 'potential',potential,potential_config_set

    def get_file_name(self,issue_type):
        if issue_type=='vanish':
            return 'gradient_vanish',self.gradient_vanish_strategy.copy()
        elif issue_type=='explode':
            return 'gradient_explode',self.gradient_explode_strategy.copy()
        elif issue_type=='relu':
            return 'dying_relu',self.dying_relu_strategy.copy()
        elif issue_type=='unstable':
            return 'training_unstable',self.unstable_strategy.copy()
        elif issue_type=='loss_issue':
            return 'training_loss_issue',self.loss_strategy.copy()
        elif issue_type=='abnormal_data':
            return 'training_abnormal_data',self.data_strategy.copy()
        elif issue_type=='activation_issue':
            return 'training_activation_issue',self.activation_strategy.copy()
        elif issue_type=='not_converge':
            return 'training_not_converge',self.not_converge_strategy.copy()
        elif issue_type=='overfit':
            return 'over_fitting',self.over_fitting_strategy.copy()
    
    
    def update_current_solution(self,issue_type,solution):
        if issue_type=='vanish':
            return self.gradient_vanish_strategy.remove(solution)
        elif issue_type=='explode':
            return self.gradient_explode_strategy.remove(solution)
        elif issue_type=='relu':
            return self.dying_relu_strategy.remove(solution)
        elif issue_type=='unstable':
            return self.unstable_strategy.remove(solution)
        elif issue_type=='not_converge':
            return self.not_converge_strategy.remove(solution)
        elif issue_type=='loss_issue':
            return self.loss_strategy.remove(solution)
        elif issue_type=='activation_issue':
            return self.activation_strategy.remove(solution)
        elif issue_type=='abnormal_data':
            return self.data_strategy.remove(solution)
        elif issue_type=='overfit':
            return self.over_fitting_strategy.remove(solution)

