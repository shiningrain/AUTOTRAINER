import os
import sys
sys.path.append('../../data')
sys.path.append('../../utils')
from cifar10 import load_data,preprocess
from utils import *
import numpy as np
import keras
from keras.optimizers import SGD,Adam
import datetime
import numpy as np
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras import backend as K
from keras.layers import Input, Conv2D, Dense, GlobalAveragePooling2D, BatchNormalization
from keras.layers import ReLU, Activation, Dropout, Reshape, Add, DepthwiseConv2D
from keras.models import Model

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class mobile_v2_model:
    """
    Refer to https://github.com/xiaochus/MobileNetV2/blob/master/mobilenet_v2.py
    """
    def __init__(self, input_shape, cls_num=10, alpha=1.0):
        self.name = 'MobileNetV2'
        self.input_shape = input_shape
        self.cls_num = cls_num
        self.alpha = alpha  #alpha: Integer, width multiplier, better in [0.35, 0.50, 0.75

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        first_filters = _make_divisible(32 * self.alpha, 8)
        x = self._conv_block(inputs, first_filters, (3, 3), strides=(2, 2))

        x = self._inverted_residual_block(x, 16, (3, 3), t=1, alpha=self.alpha, strides=1, n=1)
        x = self._inverted_residual_block(x, 24, (3, 3), t=6, alpha=self.alpha, strides=2, n=2)
        x = self._inverted_residual_block(x, 32, (3, 3), t=6, alpha=self.alpha, strides=2, n=3)
        x = self._inverted_residual_block(x, 64, (3, 3), t=6, alpha=self.alpha, strides=2, n=4)
        x = self._inverted_residual_block(x, 96, (3, 3), t=6, alpha=self.alpha, strides=1, n=3)
        x = self._inverted_residual_block(x, 160, (3, 3), t=6, alpha=self.alpha, strides=2, n=3)
        x = self._inverted_residual_block(x, 320, (3, 3), t=6, alpha=self.alpha, strides=1, n=1)

        if self.alpha > 1.0:
            last_filters = _make_divisible(1280 * self.alpha, 8)
        else:
            last_filters = 1280

        x = self._conv_block(x, last_filters, (1, 1), strides=(1, 1))
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, last_filters))(x)
        x = Dropout(0.3, name='Dropout')(x)
        x = Conv2D(self.cls_num, (1, 1), padding='same')(x)

        x = Activation('softmax', name='softmax')(x)
        output = Reshape((self.cls_num,))(x)

        mobilenet_v2 = Model(inputs, output)
        return mobilenet_v2

    def _conv_block(self, inputs, filters, kernel, strides):
        """Convolution Block
        This function defines a 2D convolution operation with BN and relu6.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
        # Returns
            Output tensor.
        """
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
        x = BatchNormalization(axis=channel_axis)(x)
        x = ReLU(max_value=6.0)(x)
        return x


    def _bottleneck(self, inputs, filters, kernel, t, alpha, s, r=False):
        """Bottleneck
        This function defines a basic bottleneck structure.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            t: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            alpha: Integer, width multiplier.
            r: Boolean, Whether to use the residuals.
        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        # Depth
        tchannel = K.int_shape(inputs)[channel_axis] * t
        # Width
        cchannel = int(filters * alpha)

        x = self._conv_block(inputs, tchannel, (1, 1), (1, 1))

        x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = ReLU(max_value=6.0)(x)

        x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)

        if r:
            x = Add()([x, inputs])
        return x

    def _inverted_residual_block(self, inputs, filters, kernel, t, alpha, strides, n):
        """Inverted Residual Block
        This function defines a sequence of 1 or more identical layers.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            t: Integer, expansion factor.
                t is always applied to the input size.
            alpha: Integer, width multiplier.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            n: Integer, layer repeat times.
        # Returns
            Output tensor.
        """

        x = self._bottleneck(inputs, filters, kernel, t, alpha, strides)

        for i in range(1, n):
            x = self._bottleneck(x, filters, kernel, t, alpha, 1, True)

        return x


if __name__ == '__main__':
    tmp=mobile_v2_model(input_shape=(32,32,3))
    model=tmp.build_model()
    save_model(model,'/data/zxy/DL_tools/DL_tools/models/seed_model/mobile_v2_seed.h5')
    model=load_model('/data/zxy/DL_tools/DL_tools/models/seed_model/mobile_v2_seed.h5')
    #config:
    labels=10
    (x, y), (x_val, y_val)=load_data()
    x=preprocess(x,'tensorflow')
    x_val=preprocess(x_val,'tensorflow')
    y = keras.utils.to_categorical(y, labels)
    y_val = keras.utils.to_categorical(y_val, labels)
    #opt='SGD'
    #opt=SGD(lr=0.1)
    opt='Adam'
    loss='categorical_crossentropy'
    dataset={}
    dataset['x']=x
    dataset['y']=y
    dataset['x_val']=x_val
    dataset['y_val']=y_val
    epoch=30
    batch_size=256
    log_dir='../../log/mobile_v2_tmp/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path=log_dir+'mobile_v2_tmp.csv'
    fig_name=log_dir+'mobile_v2_tmp.pdf'
    callbacks=[]

    model,history=model_train(model=model,optimizer=opt,loss=loss,dataset=dataset,iters=epoch,batch_size=batch_size,log_path=log_path,callbacks=callbacks,verb=1)
    gradient_list,layer_outputs,wts=gradient_test(model,dataset,batch_size)
    tmp_list=[]
    for i in range(len(gradient_list)):
        zeros=np.sum(gradient_list[i]==0)
        tmp_list.append(zeros/gradient_list[i].size)
    ave_g,_=average_gradient(gradient_list)
    issue_list=determine_issue(gradient_list,history,layer_outputs,model,threshold_low=1e-3,threshold_high=1e+3)
    #problem=gradient_issue(gradient_list)
    result_dic=read_csv(log_path,epoch)
    generate_fig(result_dic,fig_name)
    print('finish')
