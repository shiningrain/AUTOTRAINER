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

Dying_ReLU_status=0
Gradient_vanish_status=0
Gradient_explode_status=0
threshold_low=1e-3
threshold_high=1e+3

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


def trend(input_list,method='gradient',params_1=0.7,params_2=3):
    """
    the parameters comes from experience
    :params params_1: Used to determine the trend of gradient
    :params params_2: Use to determine the stable of loss/acc.
    """
    diff=[]
    positive=0
    negative=0
    # how to judge the trend of an array?
    for i in range(len(input_list)-1):
        tmp=input_list[i]-input_list[i+1]
        diff.append(tmp)
        if tmp>=0:#the previous layer has greater gradient.
            positive+=1
        if tmp<0:
            negative+=1
    if method=='gradient':
        if (input_list[0]-input_list[-1])>=0 and positive > params_1* negative:
            return 'explode'
        elif (input_list[0]-input_list[-1])<0 and negative > params_1*positive:
            return 'vanish'
        else: return 'no'
    elif method=='stable':
        tmp_min=min(input_list[0],input_list[-1])
        if positive >  params_2*negative or negative > params_2*positive:
            return 'no'
        else: return 'unstable'

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
    outputs=functor([trainingExample,0])#dense_1_input:0 is both fed and fetched.
    wts=model.get_weights()
    return outputs,wts

def gradient_test(model,dataset,batch_size):
    """
    :params model: a trained model
    :params dataset: a dictionary which contains 'x''y''x_val''y_val'
    :params batch_size: testing batch_size
    :return : gradient list.
    """
    outputTensor = model.output #Or model.layers[index].output
    listOfVariableTensors = model.trainable_weights
    #or variableTensors = model.trainable_weights[0]
    gradients = K.gradients(outputTensor, listOfVariableTensors)
    x=dataset['x']
    trainingExample = x[:batch_size,...]
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    evaluated_gradients = sess.run(gradients,feed_dict={model.input:trainingExample})

    x=dataset['x']
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
    return evaluated_gradients,outputs,wts

def average_gradient(gradient_list):
    assert len(gradient_list)%2==0
    average_kernel_list=[]
    average_bias_list=[]
    for i in range(int(len(gradient_list)/2)):
        average_kernel_list.append(np.mean(gradient_list[2*i]))
        average_bias_list.append(np.mean(gradient_list[2*i+1]))
    return average_kernel_list,average_bias_list

def tmp_acc_test(acc_list,target_acc,threshold):
    acc_start=acc_list[0]
    max_acc=np.array(acc_list).max()
    tmp=acc_start+threshold*(target_acc-acc_start)
    if acc_start>tmp: return False# start with good acc
    if max_acc<=tmp and max_acc<=3*acc_start: return True # still bad acc. improved too little
    return False

def loss_issue(feature_dic,history,total_epoch,expect_acc,\
    unstable_threshold=0.05,judgment_point=0.3,judgment_threshold=0.3,\
    not_converge_threshold=0.3):#overfit?
    """
    only consider convergence and the stability.
    2020.6.4: Only consider the training loss and acc now.a is used to evaluate the models \
        that have obtained good results in the first epoch of training. The loss and acc in the history of this model seems to be very good at the begin of training.
        when loss not converge, the curve will also have the zigzag which will interfere with our judgment
    2020.7.13 让用户设置一个acc满意区间，例如用户设定acc0.6就可以了，那我们可以训练到0.6就认为收敛。
            'not_converge':False,
            'unstable_loss':False,
            'nan_loss':False,
            'test_not_well':0,#test acc and train acc has big gap
            'test_turn_bad':0,
    """
    train_loss=history['loss']
    train_acc=history['acc']
    test_loss=history['val_loss']
    test_acc=history['val_acc']

    if train_loss!=[]:
        current_epoch=len(train_loss)#use the history to determine the current epoch
        unstable_count=0
        total_count= current_epoch-1

        if has_NaN(test_loss) or has_NaN(train_loss):
            feature_dic['nan_loss']=True
            return feature_dic
        if test_acc[-1] < expect_acc:# when this epoch is not satisfied the expect acc, we determine the issue.
            if train_acc[-1]-test_acc[-1]>=0.1:# training is far more better than testing
                feature_dic['test_not_well']+=1
            if (train_loss[-1]- train_loss[-2]>0) and (test_loss[-1]- test_loss[-2]<0):
                feature_dic['test_turn_bad']+=1
            if current_epoch>=judgment_point*total_epoch:
                for i in range(total_count):
                    if test_loss[i+1] - test_loss[i] > unstable_threshold * abs(test_loss[i]):
                        unstable_count+=1
                        if unstable_count>=judgment_threshold*total_count:
                            feature_dic['unstable_loss']=True
                if tmp_acc_test(train_acc, expect_acc,
                                not_converge_threshold) and tmp_acc_test(# or\
                                    test_acc, expect_acc, not_converge_threshold):
                    feature_dic['not_converge'] = True
    # loss_start=training_loss[0]#loss_max=training_loss.max()
    # loss_end=training_loss[-1]#loss_min=training_loss.min()
    # acc_start=training_acc[0]#acc_max=training_acc.max()
    # acc_end=training_acc[-1]#acc_min=training_acc.min()
    # if has_NaN(training_loss)or has_NaN(training_acc):
    #     loss_NaN=True
    # else:
    #     if acc_start<default_threshold:#not converge at first.
    #         if ((loss_start-loss_end)<0.8*min(loss_start,loss_end)) and ((acc_end-acc_start)<0.8*min(acc_start,acc_end)):#具体的判断需要修改
    #             loss_not_converge=True
    #     if loss_not_converge==False and trend(training_loss,method='stable',params_2=5)=='unstable':
    #         train_unstable=True
    return feature_dic

def weights_issue(feature_dic,weights,last_weights,threshold_large=5,threshold_change=0.1):
    """[summary]

    Args:
        weights ([type]): [description]
        threshold ([type]): [description]
        'large_weight':0,
        'nan_weight':False,
        'weight_change_little':0,
    """
    #max_weight=0
    change_weight_list=[]
    for i in range(len(weights)):
        if has_NaN(weights[i]):
            feature_dic['nan_weight']=True
            return feature_dic
    for i in range(len(weights)):
        if np.abs(weights[i]).max()>threshold_large:
            feature_dic['large_weight']+=1
            break
    # if np.array(weights).max()>=threshold_large:
    #     feature_dic['large_weight']+=1
        if last_weights!=[]:
            change_weight_list.append(np.average(np.abs(weights[i]-last_weights[i]))/np.average(weights[i]))# the number need to be verified
    if change_weight_list!=[] and max(change_weight_list)<=0.1: 
        feature_dic['weight_change_little']+=1
    return feature_dic


def gradient_issue(feature_dic,gradient_list,threshold_low=1e-3,threshold_high=1e+3,threshold_die=0.5):
    """
    2020.6.3: Only consider the threshold of gradient. In fact, in the cases I have reproduced, the gradient has a clear upward trend from the output back to input.\
        maybe we can consider to combine other indicator,like the trend in the gradients.
    2020.6.4: Consider the Trend of the gradient and the threshold, maybe I can find a better judgment for the Trend?
    2020.7.13：硬梯度不可取，可以根据前后梯度下降的比例，根据可训练样本规定一个比例。
            'died_relu'
            'vanish_gradient'
            'explode_gradient'
            'nan_gradient'
    """
    [avg_kernel,avg_bias,fb_gradient_rate],\
                [total_ratio,kernel_ratio,bias_ratio,max_zero]\
                    =gradient_message_summary(gradient_list)
    # assert len(gradient_list)%2==0
    # for i in range(int(len(gradient_list)/2)):
    #     try:
    #         tmp_mean=np.mean(gradient_list[2*i])
    #     except:
    #         tmp_mean=np.nan
    for i in range(len(gradient_list)):
        if has_NaN(gradient_list[i]):
            feature_dic['nan_gradient']=True
            return feature_dic
    if (fb_gradient_rate<threshold_low):# and trend(kernel)=='vanish'):
        feature_dic['vanish_gradient']+=1
    if(fb_gradient_rate>threshold_high):# and trend(kernel)=='explode') or has_NaN(gradient_list[2*i]):
        feature_dic['explode_gradient']+=1
    if (total_ratio>=threshold_die):
        feature_dic['died_relu']+=1
    return feature_dic

# def relu_issue(model):
#     """
#     2020.6.3:for the dying relu, we can consider these indicators:
#     Changes in the gradient and the proportion of dead neurons in the layer containing relu activation.
#     How to use the gradient 0 to judge this problem?
#     """
#     global Dying_ReLU_status
#     relu_position=[]
#     for i in range(len(model.layers)):
#         if 'name' in model.layers[i].get_config().keys():
#             tmp='_'+model.layers[i].get_config()['name'].split('_')[-1]
#             tmp_name=model.layers[i].get_config()['name'].replace(tmp,'')
#             if tmp_name=='re_lu':
#                 relu_position.append(i)
#         if 'activation' in model.layers[i].get_config().keys():
#             if model.layers[i].get_config()['activation']=='relu':
#                 relu_position.append(i)
#     if len(relu_position)>=0.5*len(model.layers) and Dying_ReLU_status>=5:# Now judge dying relu by the usage of relu activation and not converge.
#         return True
#     return False


def gradient_message_summary(evaluated_gradients):
    total_ratio, kernel_ratio, bias_ratio = gradient_zero_radio(
        evaluated_gradients)
    max_zero = max(kernel_ratio)
    # totolratio.append(total_ratio)
    # maxratio.append(max_zero)
    avg_kernel, avg_bias = average_gradient(evaluated_gradients)
    fb_gradient_rate = (avg_kernel[0] / avg_kernel[-1])
    return [avg_kernel, avg_bias, fb_gradient_rate], [total_ratio, kernel_ratio, bias_ratio, max_zero]


# def determine_issue(history,model,threshold_low=1e-3,threshold_high=1e+3):
#     #Input:loss,acc,gradient,wts,layer_outputs,threshold1,2,3...
#     #Output:Problem Type.
#     global Dying_ReLU_status,Gradient_explode_status,Gradient_vanish_status
#     tmp_list=[]
#     loss_not_converge=False
#     gradient_exp=False
#     gradient_van=False
#     train_unstable=False
#     relu=False
#     issue_list=[]
#     loss_not_converge,train_unstable,loss_NaN=loss_issue(history)
#     #gradient_exp,gradient_van=gradient_issue(gradient_list,threshold_low,threshold_high)
#     if Gradient_explode_status>=5: gradient_exp=True
#     if Gradient_vanish_status>=5: gradient_van=True
#     relu=relu_issue(model)

#     if gradient_van and loss_not_converge: issue_list.append('vanish')
#     if gradient_exp and (loss_not_converge,loss_NaN): issue_list.append('explode')
#     if (not (gradient_van or gradient_exp or relu)) and loss_not_converge: issue_list.append('not_converge')
#     if train_unstable: issue_list.append('unstable')
#     if relu and loss_not_converge:issue_list.append('relu')
#     return issue_list

class IssueMonitor:
    def __init__(self,total_epoch,expect_acc):
        """[summary]

        Args:
            model ([model(keras)]): [model]
            history ([dic]): [training history, include loss, val_loss,acc,val_acc]
            gradient_list ([list]): [gradient of the weights in the first batch]
        """
        self.expect_acc=expect_acc
        self.total_epoch=total_epoch
        self.issue_list=[]
        self.last_weight=[]
        self.feature={
            'not_converge':False,#
            'unstable_loss':False,##
            'nan_loss':False,#
            'test_not_well':0,#test acc and train acc has big gap
            'test_turn_bad':0,

            'died_relu':0,#
            'vanish_gradient':0,#
            'explode_gradient':0,#
            'nan_gradient':False,#

            'large_weight':0,#
            'nan_weight':False,#
            'weight_change_little':0,#
        }# for some feature, we choose to use bool to judge, but for others, may be number will be well,
        # There will be some margin, and you we control this margin with 'determine_threshold'.

        # self.light_monitor_list=['died_relu','vanish_gradient','explode_gradient','test_not_well',\
        #     'weight_change_little','unstable_loss','nan_loss']
        #     #to make the monitor work faster, use the light monitor first, if it has True issue, then, test more.



    def determine(self,model,history,gradient_list,determine_threshold=5):
        # feature update
        self.history=history
        self.gradient_list=gradient_list
        self.weights=model.get_weights()
        self.feature=gradient_issue(self.feature,self.gradient_list)
        self.feature=weights_issue(self.feature,self.weights,self.last_weight)
        self.feature=loss_issue(self.feature,self.history,total_epoch=self.total_epoch,expect_acc=self.expect_acc)
        self.last_weight=self.weights

        #issue determine.
        if self.feature['nan_loss'] or self.feature['nan_weight'] or self.feature['nan_gradient']:self.issue_list.append('explode')
        if self.feature['not_converge']:
            if (self.feature['vanish_gradient']>=determine_threshold): self.issue_list.append('vanish')
            elif (self.feature['explode_gradient']>=determine_threshold) or (self.feature['large_weight']>=determine_threshold)\
                or self.feature['unstable_loss'] :                self.issue_list.append('explode')
            elif (self.feature['died_relu']>=determine_threshold): self.issue_list.append('relu')
            elif self.feature['weight_change_little']>determine_threshold: self.issue_list.append('not_converge')
            else: self.issue_list.append('not_converge')
        if self.feature['unstable_loss'] and not self.feature['not_converge']:self.issue_list.append('unstable')
        if self.feature['test_turn_bad']>determine_threshold or self.feature['test_not_well']>determine_threshold:self.issue_list.append('overfit')
        self.issue_list=list(set(self.issue_list))
        return self.issue_list
