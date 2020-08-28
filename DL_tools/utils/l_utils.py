'''
@Author: your name
@Date: 2020-07-17 08:57:45
@LastEditTime: 2020-07-19 22:42:50
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /GradientVanish/data/zxy/DL_tools/DL_tools/utils/l_utils.py
'''
import keras
import numpy as np
import keras.backend as K
# generate 2d classification dataset
def test_history_data(history,test_num):
    data_num = len(history[0])
    new_history_data = []
    for i in range(len(history)):
        new_history_data.append(history[i][data_num-test_num:data_num])
    return new_history_data
def vanish_issue(gradient_list,params_1 = 1e3):
    '''
    params_1 is The maximum threshold of the ratio of last layer gradient to the first layer gradient
    '''
    mean_gradient_list = []
    for i in range(len(gradient_list)):
        mean_gradient_list.append(np.mean(abs(gradient_list[i])))
    gradient_ratio = mean_gradient_list[-2]/mean_gradient_list[1]
    if gradient_ratio>params_1:
        return True
    return False
def overfit_issue(training_loss,testing_loss,test_acc,expect_acc = 0.8):#?
    total_num = len(testing_loss) - 1  #为什么减一
    overfit_num = 0
    for i in range(len(testing_loss) - 1):
        if test_acc[i] < expect_acc:
            if training_loss[i] - training_loss[
                    i + 1] > 0 and testing_loss[i] - testing_loss[i + 1] < 0:
                overfit_num += 1

    print(overfit_num, total_num)
    if overfit_num == total_num:
        return True
    return False


def unstable_issue(testing_loss,testing_acc,expect_acc = 0.95,params_1 = 0.3,params_2 = 0.05):
    '''
    params_1 The threshold ratio of unstable number in the total number
    params_2 loss upper and lower range
    '''
    total_num = len(testing_loss)-1
    unstable_num = 0

    for i in range(len(testing_loss)-1):
        if testing_acc[i] < expect_acc:
            if testing_loss[i+1] - testing_loss[i] > params_2 * abs(testing_loss[i]):
                unstable_num += 1
    print(unstable_num,total_num)
    if unstable_num/total_num>params_1:
        return True
    return False
class LossHistory(keras.callbacks.Callback):

    def __init__(self,training_data,testing_data,model,batch_size = 32,expect_acc = 0.95,checktype = 'epoch_5',test_num = 5,):
        self.trainX = training_data[0]
        self.trainy = training_data[1]
        self.testX = testing_data[0]
        self.testy = testing_data[1]
        self.batch_size=batch_size
        self.model=model
        self.expect_acc = expect_acc
        self.Gradient_vanish = False
        self.not_stable = False
        self.overfitting = False
        self.test_num = test_num
        self.history = [[] for i in range(4)]
    def on_train_begin(self,logs=None):
        self.Gradient_vanish_status = 0
        weights=self.model.trainable_weights# get trainable weights
        grads = self.model.optimizer.get_gradients(self.model.total_loss, weights)
        symb_inputs = [self.model._feed_inputs , self.model._feed_targets , self.model._feed_sample_weights,K.learning_phase()]
        #input,corresponding label,weight of each sample(all of them are 1),learning rate(we set it to 0)
        self.f = K.function(symb_inputs, grads)
        print('start')
    def on_epoch_end(self,epoch,logs={}):#加入对loss和acc的观测方法
        trainingExample = self.trainX[0:self.batch_size,...]
        trainingY=self.trainy[0:self.batch_size]
        x, y, sample_weight = self.model._standardize_user_data(trainingExample, trainingY)
        #output_grad = f(x + y + sample_weight)
        output_grad = self.f([x , y , sample_weight,0])
        self.Gradient_vanish_status += vanish_issue(output_grad)
        self.history[0].append(logs.get('loss'))
        self.history[1].append(logs.get('accuracy'))
        self.history[2].append(logs.get('val_loss'))
        self.history[3].append(logs.get('val_accuracy'))
        #_history.append(train_loss,train_acc,test_loss,test_acc)
        if epoch >= self.test_num:#仅仅检测5个epoch来判断是否产生问题，加速判断
            test_history = test_history_data(self.history,self.test_num)
            print(test_history[0],test_history[2],test_history[3])
            self.overfitting = overfit_issue(test_history[0],test_history[2],test_history[3],expect_acc = self.expect_acc)
            self.not_stable = unstable_issue(test_history[2],test_history[3],expect_acc = self.expect_acc)
        if self.Gradient_vanish_status>=5:
            self.Gradient_vanish = True
        print(self.Gradient_vanish_status,self.Gradient_vanish,self.not_stable,self.overfitting)
