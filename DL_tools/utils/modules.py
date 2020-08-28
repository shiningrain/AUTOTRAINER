import os
import sys
sys.path.append('.')
import utils
import copy
import matplotlib.pyplot as plt
import csv
import numpy as np
import keras
import datetime
import repair as rp
from TimeCounter import TimeHistory
from keras.models import load_model,Sequential
import keras.backend as K
import tensorflow as tf
from logger import Logger
logger = Logger()
import time
import uuid
from collections import Counter
# from utils import model_retrain

from keras.models import load_model
from keras.models import Model
from keras.activations import relu,sigmoid,elu,linear,selu
from keras.layers import BatchNormalization,GaussianNoise
from keras.layers import Activation,Add,Dense
from keras.layers.core import Lambda
from keras.initializers import he_uniform,glorot_uniform,zeros
from keras.optimizers import SGD, Adam, Adamax
from keras.callbacks.callbacks import ReduceLROnPlateau

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
        'estop':{'modify':0, 'memory':False}
    }
problem_evaluation = {# please use the priority order here
    'vanish': 2,
    'explode': 2,
    'relu': 2,
    'not_converge': 1,
    'unstable': 1,
    'overfit': 0
}

def filtered_issue(issue_list):
    for i in range(len(issue_list)):
        if problem_evaluation[issue_list[i]]==2:
            return [issue_list[i]]
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

def read_strategy(string):
    solution=string.split('_')[0]
    times= string.split('_')[-1]
    #if string.split('_')[-1]=='': times=1
    return solution,int(times)

def get_count_by_count(l):
    # tmp_dict = {}
    # s = set(l) #集合：得到不重复的元素
    # for i in s:
    #     tmp_dict[i] = l.count(i) #对集合中每个元素分别计数，存入dictionary中
    # return tmp_dict
    count = Counter(l)   #类型： <class 'collections.Counter'>
    count_dict = dict(count)   #类型： <type 'dict'>
    return count_dict

def retrain(model,config,issue_list,save_dir,i,j,callbacks=[]):
    log_name=str(i)+'_'+str(j)+'.csv'
    log_path=os.path.join(save_dir,log_name)
    untrained_model_path=os.pth.join(save_dir,'untrained_model.h5')
    model.save(untrained_model_path)
    trained_model,history,train_result=model_retrain(model=model,optimizer=config['opt'],loss=config['loss'],dataset=config['dataset'],iters=config['epoch'],batch_size=config['batch_size'],\
        log_path=log_path,callbacks=config['callbacks'],verb=1)#simple train, only show the issue list
    new_issue_list=determine_issue(history,model,threshold_low=1e-3,threshold_high=1e+3)
    if len(new_issue_list)==0:
        effect='solved'
    elif set(issue_list).issubset(set(new_issue_list)):
        effect='no'
    else: effect='potential'
    return model,config,effect,new_issue_list,train_result

def notify_result(num, model,config,issue,j):
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
        'tanh':rp.op_tanh
    }

    method = numbers.get(num, rp.repair_default)
    if method:
        return method(model,config,issue,j)

class Repair_Module:
    def __init__(self,model,training_config,issue_list,sufferance, memory,satisfied_acc, method='balance'):
        """#pure_config,
        method:['efficiency','structure','balance'], efficient will try the most efficiently solution and the structure will
            first consider to keep the model structure/training configuration.balance is the compromise solution.
        """
        tmp_model=copy.deepcopy(model)
        #self.pure_config=pure_config
        self.satisfied_acc=satisfied_acc
        self.model=tmp_model
        self.issue_list=issue_list
        self.initial_issue=copy.deepcopy(issue_list)
        self.train_config=training_config
        self.best_potential=[]
        #if len(issue_list)>=2: self.multi=True
        #'_x' in strategy means this solution will be tried at most x times.
        strategy_list=rp.repair_strategy(method)
        self.gradient_vanish_strategy,self.gradient_explode_strategy,self.dying_relu_strategy,\
             self.unstable_strategy,self.not_converge_strategy,self.over_fitting_strategy=filtered(strategy_list,sufferance,memory)


    def solve(self,tmp_dir,retry=3): # need multi solve + retry
                #     return 'solved', [save_dir,config,model_path,log_name]
                # elif issue_type not in new_issue_list:
                #     potential[i]=[new_model,config,new_issue_list,train_result,log]
                # if log==[] or log[-1]<=train_result:
                #     log=[new_model,config,new_issue_list,train_result]
                

        if len(self.issue_list)==1:# only one issue to be solved:
            for i in range(retry):
                result,result_list=self.issue_repair_one(self.model,self.train_config,tmp_dir,self.issue_list)
                if result=='solved':
                    print('Your model has been trained and the training problems {} have been repaired,you can find the repair log in {}, and the trained model is saved in {}.'.format(self.initial_issue,result_list[-1],\
                        result_list[-2]))
                    return result,result_list[0],result_list[-2]
                elif result == 'no':
                    print(
                        'The problems {} are still exist, you can try other solutions, see the file in {}, repair log can be found in {}'
                        .format(self.initial_issue, '/path',result_list[-2]))
                    return result,result_list[0],result_list[-1]
                elif result == 'potential':
                    #[new_model,config,new_issue_list,train_result,log_name]
                    potential_list=result_list
                    self.model=potential_list[0]
                    self.issue_list=issue_list[2]
                    self.train_config=training_config[1]
                    if self.best_potential==[] or self.best_potential[-2]<potential_list[-2]:
                        self.best_potential=potential_list
                    if i==(retry-1):
                        model_name='improved_model_'+str(self.best_potential[-1])+'.h5'
                        model_path=os.path.join(tmp_dir,model_name)
                        self.best_potential[0].save(model_path)
                        print('Model has been improved but still has problem. The initial training problems in the source model is {}.\
                            The current problem is {}. You can find the repair log in {}, and the best improved model is saved \
                            in {}.'.format(self.initial_issue,self.best_potential[3],self.best_potential[-1],model_path))
                        return result,self.best_potential[0],model_path
        elif len(self.issue_list)>1:
            #[new_model,save_dir,config,model_path,log_name]
            #        return 'no',[model,train_config,issue_list,log_name]
            result,result_list=self.issue_repair_multi(self.model,self,train_config,tmp_dir,self.issue_list)
            if result=='solved':
                print('Your model has been trained and the training problems {} have been repaired,\
                        you can find the repair log in {}, and the trained model is saved in {}.'.format(self.initial_issue,result_list[-1],\
                        result_list[-2]))
                return result,result_list[0],result_list[-2]
            if result=='no':
                print('The problems {} are still exist, you can try other solutions, see the file in {}, repair log can be found in {}'
                        .format(self.initial_issue, '/path',result_list[-2]))
                return result,result_list[0],result_list[-1]
            if result=='potential':
                #[model,train_config,new_issue_list,train_result,log_name,model_path]
                print('Model has been improved but still has problem. The initial training problems in the source model is {}.\
                            The current problem is {}. You can find the repair log in {}, and the best improved model is saved \
                            in {}.'.format(self.initial_issue,result_list[3],result_list[-2],result_list[-1]))
                return result,result_list[0],result_list[-1]


    def issue_repair_multi(self,model,train_config,tmp_dir,issue_list,retry=3):#need modify
        """[summary]

        Args:
            tmp_dir ([type]): [description]
            issue_type (bool): [description]
            加入解决次数上限设计
        Returns:
            [type]: [description]
        """
        total_solution_list=[]
        tmp_list=[]
        solution_list={}# this is a dic
        for issue_type in issue_list:
            _,solution_list[issue_type]=self.get_file_name(issue_type)
            for i in range(len(solution_list[issue_type])):
                total_solution_list.append(solution_list[issue_type][i])
        log_name='{}_{}.log'.format('multi_issue',str(time.time()))
        log_name=os.path.join(tmp_dir,log_name)
        log_file=open(log_name,'a+')
        log_file.write('solution,times,issue_list,train_result,describe\n')
        start_time=time.time()
        potential={}
        issue_number=len(issue_list)
        to_solved_issue=[]
        tmp_solved_list=[]
        for key in problem_evaluation:
            if key in issue_list:
                to_solved_issue.append[key]
        for issue in range(issue_number):
            now_issue=to_solved_issue[issue]
            now_solution=solution_list[now_issue]
            for i in range(len(now_solution)):# Try common solutions first
                tmp_sol,tmp_tim=read_strategy(now_solution[i])
                for j in range(tmp_tim):
                    _break=False
                    tmp_model,config,modify_describe,_break=notify_result(tmp_sol,model,train_config,issue_list,j)
                    if _break: break#  the solution has already been used in the source model.
                    solution_dir='multi_'+str(tmp_sol)+'_'+str(j)
                    save_dir=os.path.join(tmp_dir,solution_dir)
                    new_model,new_issue_list,train_result=utils.model_retrain(tmp_model,config,save_dir,satisfied_acc=self.satisfied_acc)
                    log_file.write('{},{},{},{},{}\n'.format(tmp_sol,j,new_issue_list,train_result,describe))
                    log_file.flush()
                    if new_issue_list==[]:
                        end_time=time.time()
                        time_used=end_time-start_time
                        log_file.close()
                        # how to save config?
                        # use describe, we will describe how we modified user's configuration.
                        model_name='solved_model_'+str(train_result)+'.h5'
                        model_path=os.path.join(save_dir,model_name)
                        new_model.save(model_path)
                        return 'solved', [new_model,save_dir,config,model_path,log_name]
                    elif set(new_issue_list)<set(issue_list) and (now_issue not in new_issue_list):
                        model=new_model
                        train_config=config
                        tmp_solved_list=[model,train_config,new_issue_list,train_result,log_name]
                        break
                break
            continue
        end_time=time.time()
        time_used=end_time-start_time
        log_file.close()
        print('------------------Unsolved..Time used {}!-----------------'.format(str(time_used)))
        if tmp_solved_list!=[]:
            model_name='unsolved_model_'+str(tmp_solved_list[-2])+'.h5'
            model_path=os.path.join(save_dir,model_name)
            tmp_solved_list.save(model_path)
            tmp_solved_list.append(model_path)
            return 'potential',tmp_solved_list
        model_name='unsolved_model_'+str(log[-1])+'.h5'
        model_path=os.path.join(save_dir,model_name)
        model.save(model_path)
        return 'no',[model,train_config,issue_list,log_name,model_path]


    def issue_repair_one(self,model,train_config,tmp_dir,issue_list):#need modify
        """[summary]

        Args:
            tmp_dir ([type]): [description]
            issue_type (bool): [description]
            加入解决次数上限设计
        Returns:
            [type]: [description]
        """
        #file_name,solution_list
        #if len(issue_list)==1:
        issue_type=issue_list[0]
        file_name,solution_list=self.get_file_name(issue_type)
        log_name='{}_{}.log'.format(file_name,str(time.time()))
        log_name=os.path.join(tmp_dir,log_name)
        log_file=open(log_name,'a+')
        log_file.write('solution,times,issue_list,train_result,describe\n')
        start_time=time.time()
        potential={}
        log=[]
        for i in range(len(solution_list)):
            tmp_sol,tmp_tim=read_strategy(solution_list[i])
            for j in range(tmp_tim):
                # solutions
                _break=False
                tmp_model,config,modify_describe,_break=notify_result(tmp_sol,model,train_config,issue_type,j)
                if _break: break#  the solution has already been used in the source model.
                solution_dir=str(issue_type)+'_'+str(tmp_sol)+'_'+str(j)
                save_dir=os.path.join(tmp_dir,solution_dir)
                new_model,new_issue_list,train_result=utils.model_retrain(tmp_model,config,satisfied_acc=self.satisfied_acc,retrain_dir=save_dir)
                log_file.write('{},{},{},{},{}\n'.format(tmp_sol,j,new_issue_list,train_result,modify_describe))
                log_file.flush()
                if new_issue_list==[]:
                    end_time=time.time()
                    time_used=end_time-start_time
                    log_file.close()
                    #how to save config?
                    model_name='solved_model_'+str(train_result)+'.h5'
                    model_path=os.path.join(save_dir,model_name)
                    new_model.save(model_path)
                    print('------------------Solved! Time used {}!-----------------'.format(str(time_used)))
                    return 'solved', [new_model,save_dir,config,model_path,log_name]
                elif issue_type not in new_issue_list:
                    potential[i]=[new_model,config,new_issue_list,train_result,log_name]
                if log==[] or log[-2]<=train_result:
                    log=[new_model,config,new_issue_list,train_result,log_name]
        log_file.close()
        if potential=={}: 
            end_time=time.time()
            time_used=end_time-start_time
            model_name='unsolved_model_'+str(log[-2])+'.h5'
            model_path=os.path.join(save_dir,model_name)
            log[0].save(model_path)
            log.append(model_path)
            print('------------------Unsolved..Time used {}!-----------------'.format(str(time_used)))
            return 'no',log
        else:
            max_strategy=None
            for key in potential:
                if max_strategy==None:
                    max_strategy=key
                elif potential[key][-2]>=potential[max_strategy][-2]:
                    max_strategy=key
            end_time=time.time()
            time_used=end_time-start_time
            # model_name='unsolved_model_'+str(potential[max_strategy][-1])+'.h5'
            # model_path=os.path.join(save_dir,model_name)
            # new_model.save(model_path)
            # potential[max_strategy].append(model_path)
            print('------------------Unsolved..Time used {}!-----------------'.format(str(time_used)))
            return 'potential',potential[max_strategy]

    def get_file_name(self,issue_type):
        if issue_type=='vanish':
            return 'gradient_vanish',self.gradient_vanish_strategy
        elif issue_type=='explode':
            return 'gradient_explode',self.gradient_explode_strategy
        elif issue_type=='relu':
            return 'dying_relu',self.dying_relu_strategy
        elif issue_type=='unstable':
            return 'training_unstable',self.unstable_strategy
        elif issue_type=='not_converge':
            return 'training_not_converge',self.not_converge_strategy
        elif issue_type=='overfit':
            return 'over_fitting',self.over_fitting_strategy
#zxy:完善代码，进行overfit实验
#lj：overfit原理分析
#考虑综合评价方法：可否使用一个问题的几个特征在应用解决方案后还剩几个的方法评测解决效果。
#考虑实验中对每个方案解决问题的能力（解决/总数）进行统计，进一步优化相关的配置。