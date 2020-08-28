import sys
sys.path.append('../../data')
sys.path.append('../../utils')
sys.path.append('../../modules')
from cifar10 import load_data,preprocess
from utils import *
from modules import *
import numpy as np
import keras
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD,Adam
from keras.initializers import RandomUniform
from keras import backend as K

stringx='case16_cnn(8conv_sigmoid_batch_except_last)_cifar'
labels=10

(x, y), (x_val, y_val)=load_data()
x=preprocess(x,'tensorflow')
x_val=preprocess(x_val,'tensorflow')
y = keras.utils.to_categorical(y, labels)
y_val = keras.utils.to_categorical(y_val, labels)

x=x[:10000,:,:,:]
y=y[:10000,:]
x_val=x_val[:2000,:,:,:]
y_val=y_val[:2000,:]

opt='Adam'
loss='categorical_crossentropy'
dataset={}
dataset['x']=x
dataset['y']=y
dataset['x_val']=x_val
dataset['y_val']=y_val
epoch=20
batch_size=256
log_dir='../../log/case16_cnn/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path=log_dir+'case16_cnn.csv'
fig_name=log_dir+'case16_cnn.pdf'
callbacks=[]
model_path='/data/zxy/DL_tools/DL_tools/models/GradientVanish/case16_cnn.h5'

config={}
config['opt']=opt
config['loss']=loss
config['dataset']=dataset
config['epoch']=epoch
config['batch_size']=batch_size

model = Sequential()
#init = RandomUniform(minval=0, maxval=1)
init='glorot_uniform'
model.add(Conv2D(filters = 16,
          kernel_size = (3, 3),
          padding = 'same',
          input_shape = (32, 32, 3),
          activation = 'sigmoid',kernel_initializer=init))
#model.add(BatchNormalization())
model.add(Conv2D(filters = 16,
          kernel_size = (3, 3),
          padding = 'same',
          activation = 'sigmoid',kernel_initializer=init))
#model.add(BatchNormalization())
model.add(Conv2D(filters = 16,
          kernel_size = (3, 3),
          padding = 'same',
          activation = 'sigmoid',kernel_initializer=init))
#model.add(BatchNormalization())
model.add(Conv2D(filters = 16,
          kernel_size = (3, 3),
          padding = 'same',
          activation = 'sigmoid',kernel_initializer=init))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
#建立第二个卷积层， filters 卷积核个数 36 个，kernel_size 卷积核大小 3*3
#padding 是否零填充 same 表示填充， activation 激活函数 sigmoid
model.add(Conv2D(filters = 36,
                kernel_size = (3, 3),
                padding = 'same',
                 activation='sigmoid',kernel_initializer=init))
#model.add(BatchNormalization())
model.add(Conv2D(filters = 36,
                kernel_size = (3, 3),
                padding = 'same',
                 activation='sigmoid',kernel_initializer=init))
#model.add(BatchNormalization())
model.add(Conv2D(filters = 36,
                kernel_size = (3, 3),
                padding = 'same',
                 activation='sigmoid',kernel_initializer=init))
#model.add(BatchNormalization())
model.add(Conv2D(filters = 36,
                kernel_size = (3, 3),
                padding = 'same',
                 activation='sigmoid',kernel_initializer=init))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
#加入Dropout避免过度拟合
model.add(Dropout(0.25))
#建立平坦层，将多维向量转化为一维向量
model.add(Flatten())
#建立隐藏层，隐藏层有 128 个神经元， activation 激活函数用 sigmoid
model.add(Dense(128, activation = 'sigmoid',kernel_initializer=init,bias_initializer='zeros'))#,
#model.add(BatchNormalization())
#加入Dropout避免过度拟合
#model.add(Dropout(0.25))
#建立输出层，一共有 10 个神经元，因为 0 到 9 一共有 10 个类别， activation 激活函数用 softmax 这个函数用来分类
model.add(Dense(10, activation = 'softmax'))

print(model.summary())
#save_model(model,model_path)
#trained_model,history,_=model_train(model=model,optimizer=opt,loss=loss,dataset=dataset,iters=epoch,batch_size=batch_size,log_path=log_path,callbacks=callbacks,verb=1)
#issue_list=determine_issue(history,trained_model,threshold_low=1e-3,threshold_high=1e+3)
issue_list=['vanish']
rm=Repair_Module(model,config,issue_list)
result=rm.solve()
#evaluated_gradients,outputs,wts=gradient_test(trained_model,dataset,batch_size)
#avg=average_gradient(evaluated_gradients)
#result_dic=read_csv(log_path,epoch)
#generate_fig(result_dic,fig_name)