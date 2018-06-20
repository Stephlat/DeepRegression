# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
'''
from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
import os.path



WEIGHTS_PATH = '/pathtoweights/resnet50_weights_th_dim_ordering_th_kernels.h5'

print(WEIGHTS_PATH)

def identity_block(input_tensor, kernel_size, filters, stage, block,trainable=True):
    '''The identity_block is the block that has no conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a',trainable=trainable)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a',trainable=trainable)(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b',trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b',trainable=trainable)(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c',trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c',trainable=trainable)(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2),trainable=True):
    '''conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a',trainable=trainable)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a',trainable=trainable)(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b',trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b',trainable=trainable)(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c',trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c',trainable=trainable)(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1',trainable=trainable)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1',trainable=trainable)(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None,trainable=[True]*4,changePool=False):
    '''Instantiate the ResNet50 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. xput of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''

    t1,t2,t3,t4=trainable
    
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1),trainable=t1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b',trainable=t1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c',trainable=t1)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a',trainable=t2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b',trainable=t2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c',trainable=t2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d',trainable=t2)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a',trainable=t3)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b',trainable=t3)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c',trainable=t3)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d',trainable=t3)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e',trainable=t3)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f',trainable=t3)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a',trainable=t4)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b',trainable=t4)
    xlast = identity_block(x, 3, [512, 512, 2048], stage=5, block='c',trainable=t4)

    if changePool=="max":
        x = MaxPooling2D((7, 7), name='avg_pool')(xlast)
    else:
        x = AveragePooling2D((7, 7), name='avg_pool')(xlast)

    if include_top:
        xflat = Flatten()(x)
        x = Dense(1000, activation='softmax', name='fc1000')(xflat)

    model = Model(img_input, x)

    # load weights
    model.load_weights(WEIGHTS_PATH)
    if K.backend() == 'theano':
        convert_all_kernels_in_model(model)
    if changePool== "none":
        xflatNone = Flatten()(xlast)
        modelout = Model(img_input, xflatNone, name='resnet50')
    else:
        modelout = Model(img_input, xflat, name='resnet50')
    
    return modelout
