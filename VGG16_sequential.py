'''VGG16 model for Keras.
# Reference:
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
'''

# from __future__ import print_function
# from __future__ import absolute_import

import warnings

import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, activations
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import socket
import os.path


WEIGHTS_PATH = '/pathTOWeights/vgg16_weights_init.h5'



# TH_WEIGHTS_PATH_DEEP_GLLIM = 'path/to/your_th_weights'
# TF_WEIGHTS_PATH_DEEP_GLLIM = 'path/to/your_tf_weights'
# TH_WEIGHTS_PATH_DEEP_GLLIM_PCA_BN = '/services/scratch/perception/dataBiwi/Deep_Gllim_pose86407_K2_weights.hdf5'

def VGG16(weights='imagenet'):
    '''Instantiate the VGG16 architecture,
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
        weights: one of `deep_gllim` (fine tunned weights)
            or "imagenet" (pre-training on ImageNet).
    # Returns
        A Keras model instance.
    '''
    # if weights not in {'imagenet', 'deep_gllim'}:
    #     raise ValueError('The `weights` argument should be either '
    #                      '`imagenet` (pre-training on ImageNet)'
    #                      'or `deep_gllim` (fine tunned weights).')

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        INPUT_SHAPE = (3, 224, 224)
    else:
        INPUT_SHAPE = (224, 224, 3)

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=INPUT_SHAPE))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
        
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', trainable=True))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', trainable=True))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax', trainable=True))
        
    # load weights
    if weights == 'imagenet':
        if K.image_dim_ordering() == 'th':
            print "LOAD: " + WEIGHTS_PATH
            weights_path = WEIGHTS_PATH
            model.load_weights(weights_path)
            model.pop() # remove softmax layer
            model.pop() # remove dropout
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
           
        else:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    TF_WEIGHTS_PATH,
                                    cache_subdir='models')
            model.load_weights(weights_path)
            model.pop() # remove softmax layer
            model.pop() # remove dropout
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)
       
         
    elif weights == 'deep_gllim':
        if K.image_dim_ordering() == 'th':
            weights_path = TH_WEIGHTS_PATH_DEEP_GLLIM
            model.load_weights(weights_path)
            model.pop() # remove softmax layer
            model.pop() # remove dropout
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
                
        else:
            weights_path = TF_WEIGHTS_PATH_DEEP_GLLIM
            model.load_weights(weights_path)
            model.pop() # remove softmax layer
            model.pop() # remove dropout
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)
    elif weights == 'deep_gllim_PCA_BN':
        model.pop()  # remove softmax layer
        model.pop()  # remove dropout
        model.add(Dense(512, activation='linear', trainable=False))
        model.add(BatchNormalization())

        weights_path = TH_WEIGHTS_PATH_DEEP_GLLIM_PCA_BN 
        model.load_weights(weights_path)
        model.pop()  # remove BN
    return model

def extract_features_generator(network, generator, size):
    '''Extract VGG features from a generator'''
    
    print("Extracting features :")
    
    features = network.predict_generator(generator, val_samples=size)
    
    return features

def extract_features(network, x):
    '''Extract VGG features from a generator'''
    
    print("Extracting features :")
    
    features = network.predict(x, batch_size=64)
    
    return features

def extract_XY_generator(network, generator, size):
    '''Extract VGG features and data targets from a generator'''
    
    i=0
    X=[]
    Y=[]
    for x,y in generator:
        X.extend(network.predict_on_batch(x))
        Y.extend(y)
        i+=len(y)
        if i>=size:
            break
        
    return np.asarray(X), np.asarray(Y)

