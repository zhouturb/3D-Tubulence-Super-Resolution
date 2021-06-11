# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, Activation, Add, Lambda, Conv3DTranspose
from tensorflow.keras.models import Model
from pixel_shuffler import PixelShuffler3D


def res_block(input_tensor, filters, scale=0.1):
    x = Conv3D(filters=filters, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = Activation('relu')(x)

    x = Conv3D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    if scale:
        x = Lambda(lambda t: t * scale)(x)
    x = Add()([x, input_tensor])

    return x


def sub_pixel_Conv2D(scale=2, **kwargs):
    ''' 
    注：tensorflow1.x版本为tf.depth_to_space
        tensorflow2.x版本为tf.nn.depth_to_space
        此函数只适用于4D tensor.
    '''
    
    return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)


def sub_pixel_Conv3D(scale=2):
    return Conv3DTranspose(16, 3, scale, padding='same')

def upsample(input_tensor, filters):
    x = Conv3D(filters=filters*4, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = sub_pixel_Conv3D(scale=2)(x)
    x = Activation('relu')(x)
    return x


def generator(input_shape, filters=128, n_id_block=8, n_sub_block=3):
    inputs = Input(shape=input_shape)

    x = x_1 = Conv3D(filters=filters, kernel_size=3, strides=1, padding='same')(inputs)

    for _ in range(n_id_block):
        x = res_block(x, filters=filters)

    x = Conv3D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    x = Add()([x_1, x])

    for _ in range(n_sub_block):
        x = upsample(x, filters)
    x = Conv3D(filters=1, kernel_size=3, strides=1, padding='same')(x)

    return Model(inputs=inputs, outputs=x)

autoencoder = generator(input_shape=(16,16,16,1))
autoencoder.summary()