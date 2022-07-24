import os
import sys
sys.path.append('.')
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import datetime
from TimeCounter import TimeHistory
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf
from logger import Logger
logger = Logger()
import tensorflow.keras.backend as K
import copy
import pickle
import uuid




def inclusion_ratio(x,nparray):
    return float(np.sum(nparray==x))/float(nparray.size)

def has_NaN(output):
    output=np.array(output)
    result=(np.isnan(output).any() or np.isinf(output).any())
    return result

def gradient_zero_radio(gradient_list):
    kernel=[]
    bias=[]
    total_zero=0
    total_size=0
    for i in range(len(gradient_list)):    
        zeros=np.sum(gradient_list[i]==0)
        total_zero+=zeros
        total_size+=gradient_list[i].size
        if i%2==0:
            kernel.append(zeros/gradient_list[i].size)
        else: bias.append(zeros/gradient_list[i].size)
    total=float(total_zero)/float(total_size)
    return total,kernel,bias

def get_layer_output(model, inputs):
    """ Gets a layer output for given inputs and outputs"""
    # inputs = x[:batch_size,...]
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    outputs = []
    for layer in model.layers:
        f = K.function([model.input], [layer.output])
        outputs.append(f([inputs,  K.learning_phase()]))
    return outputs

def get_weights(model,x,batch_size):
    trainingExample = x[:batch_size,...]
    inp = model.input
    layer_outputs = []
    if model.layers[0].get_config()['name'].split('_')[0]=='input':
        for layer in model.layers[1:]:
            layer_outputs.append(layer.output)
    else:
        for layer in model.layers[0:]:
            layer_outputs.append(layer.output)
    functor = K.function([inp] + [K.learning_phase()], layer_outputs)
    outputs=functor([trainingExample,0])
    wts=model.get_weights()
    return outputs,wts


def max_delta_acc(acc_list):
    max_delta=0
    for i in range(len(acc_list)-1):
        if acc_list[i+1]-acc_list[i]>max_delta:
            max_delta=acc_list[i+1]-acc_list[i]
    return max_delta

def gradient_norm(gradient_list):
    assert len(gradient_list)%2==0
    norm_kernel_list=[]
    norm_bias_list=[]
    for i in range(int(len(gradient_list)/2)):
        # average_kernel_list.append(np.mean(np.abs(gradient_list[2*i])))
        # average_bias_list.append(np.mean(np.abs(gradient_list[2*i+1])))
        norm_kernel_list.append(np.linalg.norm(np.array(gradient_list[2*i])))
        norm_bias_list.append(np.linalg.norm(np.array(gradient_list[2*i+1])))
    return norm_kernel_list,norm_bias_list


def ol_judge(history,threshold,rate):
    acc=history['acc']
    maximum=[]
    minimum=[]
    count=0
    for i in range(len(acc)):
        if i==0 or i ==len(acc)-1:
            continue
        if acc[i]-acc[i-1]>=0 and acc[i]-acc[i+1]>=0:
            maximum.append(acc[i])
        if acc[i]-acc[i-1]<=0 and acc[i]-acc[i+1]<=0:
            minimum.append(acc[i])
    for i in range(min(len(maximum),len(minimum))):
        if maximum[i]-minimum[i]>=threshold:
            count+=1
    if count>=rate*len(acc):
        return True
    else:
        return False

def check_large_loss(test_loss,train_loss,threshold=1e+3):
    if max(test_loss)>=threshold or max(train_loss)>=threshold:
        return True
    return False

def check_abnormal_value(test_loss,train_loss):
    if min(test_loss)<0 or min(train_loss)<0:
        return True
    return False

def layer_issue(
    feature_dict,
    layer_output_list,
    input_threshold=10,
    acti_threshold=1):
    
    output_layer=layer_output_list[-1]
    other_layer=copy.deepcopy(layer_output_list)
    del other_layer[-1]
    
    sum_output_layer=np.absolute(output_layer[0]).sum(1)
    if max(sum_output_layer)>(acti_threshold+0.1):
        feature_dict['activation_output']=True
        
    for other in other_layer:
        mean_layer=np.mean(np.abs(other))
        if mean_layer>=input_threshold and not feature_dict['activation_output']:
            feature_dict['abnormal_output']=True
            break
        
    return feature_dict
    
    

def loss_issue(feature_dict,history,total_epoch,satisfied_acc,checkgap,\
    unstable_threshold=0.05,judgment_point=0.3,unstable_rate=0.25,epsilon=10e-3,sc_threshold=0.01):

    train_loss=history['loss']
    train_acc=history['acc']
    test_loss=history['val_loss']
    test_acc=history['val_acc']
    count=0

    if train_loss!=[]:
        if has_NaN(test_loss) or has_NaN(train_loss) or check_large_loss(test_loss,train_loss,1e+3):
            feature_dict['nan_loss']=True
            return feature_dict
        if check_abnormal_value(train_loss,test_loss):
            feature_dict['abnormal_loss']=True
        current_epoch=len(train_loss)
        unstable_count=0
        total_count= current_epoch-1
        if current_epoch>=judgment_point*total_epoch-checkgap:
            
            if (train_acc[-1]<=0.9 and train_acc[-1]-test_acc[-1]>=0.1)\
                or (train_acc[-1]>0.9 and train_acc[-1]-test_acc[-1]>=0.07):
                feature_dict['test_not_well']+=1
            bad_count=0

            for i in range(total_count):            
                if ((train_loss[-i-1]- train_loss[-i-2])<-epsilon) and ((test_loss[-i-1]- test_loss[-i-2])>epsilon):#train decrease and test increase.
                    bad_count+=1
                else:
                    feature_dict['test_turn_bad']=bad_count###-------------------------用overfit 例子做个检查
                    break

            if ol_judge(history,unstable_threshold,unstable_rate)==True:  
                feature_dict['unstable_loss']=True
            if max(test_acc)<satisfied_acc and max(train_acc)<satisfied_acc:
                feature_dict['not_converge'] = True
            if max_delta_acc(test_acc)<sc_threshold and max_delta_acc(train_acc)<sc_threshold:
                feature_dict['sc_accuracy']=True

    return feature_dict

def weights_issue(feature_dict,weights,last_weights,threshold_large=5,threshold_change=0.1):
    """[summary]

    Args:
        weights ([type]): [description]
        threshold ([type]): [description]
        'large_weight':0,
        'nan_weight':False,
        'weight_change_little':0,
    """
    for i in range(len(weights)):
        if has_NaN(weights[i]):
            feature_dict['nan_weight']=True
            return feature_dict
    for j in range(len(weights)):
        if np.abs(weights[j]).max()>threshold_large:
            feature_dict['large_weight']+=1
            break
    return feature_dict


def gradient_issue(feature_dict,gradient_list,threshold_low=1e-3,threshold_low_1=1e-4,threshold_high=70,threshold_die_1=0.7):

    [norm_kernel,avg_bias,gra_rate],\
                [total_ratio,kernel_ratio,bias_ratio,max_zero]\
                    =gradient_message_summary(gradient_list)


    for i in range(len(gradient_list)):
        if has_NaN(gradient_list[i]):
            feature_dict['nan_gradient']=True
            return feature_dict

    if (gra_rate<threshold_low and norm_kernel[0]<threshold_low_1):
        feature_dict['vanish_gradient']+=1
    if(gra_rate>threshold_high):
        feature_dict['explode_gradient']+=1
    if (total_ratio>=threshold_die_1):# or max_zero>=threshold_die_2
        feature_dict['died_relu']+=1
    return feature_dict



def gradient_message_summary(gradient_list):
    total_ratio, kernel_ratio, bias_ratio = gradient_zero_radio(
        gradient_list)
    max_zero = max(kernel_ratio)

    norm_kernel, norm_bias = gradient_norm(gradient_list)
    gra_rate = (norm_kernel[0] / norm_kernel[-1])
    return [norm_kernel, norm_bias, gra_rate], [total_ratio, kernel_ratio, bias_ratio, max_zero]

def loss_function_issue(feature_dict,loss_function,output_shape,activation):
    feature_dict['improper_loss']=False
    
    if isinstance(loss_function,str):
        # if activation=='sigmoid' and 'binary' in loss_function:
        #     feature_dict['improper_loss']=False
        # elif activation=='softmax' and 'categorical' in loss_function:
        #     feature_dict['improper_loss']=False
        # elif activation=='linear' and 'mean' in loss_function:
        #     feature_dict['improper_loss']=False
        if output_shape==1 and 'binary' in loss_function:
            feature_dict['improper_loss']=False
        elif output_shape>1 and 'categorical' in loss_function:
            feature_dict['improper_loss']=False
        else:
            feature_dict['improper_loss']=True
        if not feature_dict['improper_loss']:
            if activation!='sigmoid' and 'binary' in loss_function:
                feature_dict['activation_output']=True
            elif activation!='softmax' and 'categorical' in loss_function:
                feature_dict['activation_output']=True
            elif activation!='linear' and 'mean' in loss_function:
                feature_dict['activation_output']=True
    return feature_dict

class IssueMonitor:
    def __init__(self,total_epoch,satisfied_acc,params,determine_threshold=1):
        """[summary]

        Args:
            model ([model(keras)]): [model]
            history ([dic]): [training history, include loss, val_loss,acc,val_acc]
            gradient_list ([list]): [gradient of the weights in the first batch]
        """
        self.selective_feature=['abnormal_output','activation_output']
        self.satisfied_acc=satisfied_acc
        self.total_epoch=total_epoch
        self.determine_threshold=determine_threshold
        self.issue_list=[]
        self.last_weight=[]
        self.feature={
            'not_converge':False,#
            'unstable_loss':False,##
            'nan_loss':False,#
            'abnormal_loss':False,
            # 'improper_loss':False,
            'test_not_well':0,#test acc and train acc has big gap
            'test_turn_bad':0,
            # 'not_trained_well':0,
            'sc_accuracy':False,

            'died_relu':False,#
            'vanish_gradient':False,#
            'explode_gradient':False,#
            'nan_gradient':False,#
        

            'large_weight':0,#
            'nan_weight':False,#
            'weight_change_little':0,#
        }
        self.params=params
        self.initial_feature=copy.deepcopy(self.feature)
        # default_param = {'beta_1': 1e-3,
                        #  'beta_2': 1e-4,
                        #  'beta_3': 70,
                        #  'gamma': 0.7,
                        #  'zeta': 0.03,
                        #  'eta': 0.2,
                        #  'delta': 0.01,
                        #  'alpha_1': 0,
                        #  'alpha_2': 0,
                        #  'alpha_3': 0,
                        #  'Theta': 0.7
                        #  }


    def determine(self,model,history,gradient_list,checkgap,layer_output=None):
        
        #no issue model should has train or test acc better than satisfyied acc and no unstable.
        self.history=history
        self.gradient_list=gradient_list
        self.weights=model.get_weights()
        if 'improper_loss' not in self.feature.keys():
            loss_function=model.loss
            output_shape=model.output.shape[-1]
            try:
                activation_function=model.layers[-1].get_config()['activation']
            except:
                activation_function=model.layers[-1].get_config()['name'].split('_')[0]
            self.feature=loss_function_issue(self.feature,loss_function,output_shape,activation_function)
        
        if layer_output!=None:
            self.feature=layer_issue(self.feature,layer_output,input_threshold=self.params['omega_1'],acti_threshold=self.params['omega_2'])
        self.feature=gradient_issue(self.feature,self.gradient_list,threshold_low=self.params['beta_1'],threshold_low_1=self.params['beta_2'],
        threshold_high=self.params['beta_3'],threshold_die_1=self.params['gamma'])
        self.feature=weights_issue(self.feature,self.weights,self.last_weight)
    #     loss_issue(feature_dict,history,total_epoch,satisfied_acc,checkgap,\
    # unstable_threshold=0.05,judgment_point=0.3,unstable_rate=0.25,epsilon=10e-3,sc_threshold=0.01):
        self.feature=loss_issue(self.feature,self.history,total_epoch=self.total_epoch,satisfied_acc=self.params['Theta'],checkgap=checkgap,
        unstable_threshold=self.params['zeta'],unstable_rate=self.params['eta'],sc_threshold=self.params['delta'])
        self.last_weight=self.weights

        #issue determine.

        #定义一个新的feature，针对没达到目标acc。
        # if self.feature['not_trained_well']:
        #     self.issue_list=determine_training(self.feature,self.initial_feature)
        
        self.judge_issue(layer_output)            
        return self.issue_list

    def judge_issue(self,layer_output):
        if layer_output!=None and self.issue_list==[]:
            if self.feature['abnormal_output']:
                self.issue_list.append('abnormal_data')
            if self.feature['nan_loss'] or self.feature['abnormal_loss']:
                # TODO: improve loss issue judgement
                if self.feature['abnormal_loss']:
                    self.issue_list.append('loss_issue')
                elif self.feature['nan_weight'] or self.feature['nan_gradient']:
                    self.issue_list.append('explode')
            if self.feature['not_converge']  or self.feature['sc_accuracy']:    
            #备选： 每次检测最后一个acc，如果没有到目标acc，那么我们都要检测消失爆炸die。
                if self.feature['activation_output']:
                    self.issue_list.append('activation_issue')
                elif (self.feature['died_relu']>=self.determine_threshold): 
                    self.issue_list.append('relu')
                elif (self.feature['explode_gradient']>=self.determine_threshold):
                    #  or (self.feature['large_weight']>=self.determine_threshold):
                    self.issue_list.append('explode')
                elif (self.feature['vanish_gradient']>=self.determine_threshold): 
                    self.issue_list.append('vanish')
                elif (self.feature['not_converge'] and self.feature['sc_accuracy']) and not self.feature['abnormal_output']: 
                    self.issue_list.append('not_converge')
            # if self.feature['test_turn_bad'] + self.feature['test_not_well']>self.determine_threshold:
            #     self.issue_list.append('overfit')
            if self.feature['unstable_loss']:
                if self.feature['activation_output']:
                    self.issue_list.append('activation_issue')
                else:
                    self.issue_list.append('unstable')
            if not self.feature['activation_output'] and self.feature['improper_loss']:
                self.issue_list.append('loss_issue')
            
            self.issue_list=list(set(self.issue_list))
            if self.issue_list!=[]:
                print(1)
                # TODO: let abnormal be the first, with the highest priority.
        elif layer_output==None and self.feature['nan_loss']:
            if self.feature['nan_weight'] or self.feature['nan_gradient']:
                self.issue_list.append('explode')