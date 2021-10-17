from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, AveragePooling1D
from keras.layers.pooling import GlobalAveragePooling2D,GlobalAveragePooling1D
from keras.layers import Input, merge, Flatten
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
from keras.layers import Conv1D,Conv2D, MaxPooling1D


init_form = 'RandomUniform'
learning_rate = 0.001
filter_size_ori = 1
dropout_rate = 0.2
dropout_dense = 0.3
weight_decay = 0.0001

def CNN_module(input_x, name):
    # siamese network
    # model_input = Input(shape=img_dim)
    # Initial convolution

    x = Conv1D(32, filter_size_ori,
               init=init_form,
               activation='relu',
               border_mode='same',
               name='{:s}_1_conv1D'.format(name),
               bias=False,
               W_regularizer=l2(weight_decay))(input_x)
    x = MaxPooling1D(pool_size=0.3, padding='same')(x)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(64, filter_size_ori,
               init=init_form,
               activation='relu',
               border_mode='same',
               name='{:s}_2_conv1D'.format(name),
               bias=False,
               W_regularizer=l2(weight_decay))(x)
    x = MaxPooling1D(pool_size=0.3, padding='same')(x)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(128, filter_size_ori,
               init=init_form,
               activation='relu',
               border_mode='same',
               name='{:s}_3_conv1D'.format(name),
               bias=False,
               W_regularizer=l2(weight_decay))(x)
    x = MaxPooling1D(pool_size=0.3, padding='same')(x)
    x = Dropout(dropout_rate)(x)

    return x


def predition_module(x_ori, x_var):

    x = concatenate([x_ori, x_var])

    x = Flatten()(x)

    x = Dense(128,
              name='Dense_1',
              activation='relu', init=init_form,
              W_regularizer=l2(weight_decay),
              b_regularizer=l2(weight_decay))(x)

    x = Dropout(dropout_dense)(x)

    x = Dense(64,
              name='Dense_2',
              activation='relu', init=init_form,
              W_regularizer=l2(weight_decay),
              b_regularizer=l2(weight_decay))(x)

    x = Dropout(dropout_dense)(x)

    # softmax
    x = Dense(32,
              name='Dense_softmax',
              activation='softmax', init=init_form,
              W_regularizer=l2(weight_decay),
              b_regularizer=l2(weight_decay))(x)

    return x


