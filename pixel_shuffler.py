# -*- coding: utf-8 -*-

import tensorflow.keras.backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.keras.layers import Input, Lambda

    
class PixelShuffler3D(Layer):
    def __init__(self, size=2):
        super(PixelShuffler3D, self).__init__()
        self.size = size

    def call(self, inputs):
        input_shape = K.shape(inputs)
        h, w, d, c = input_shape[1], input_shape[2], input_shape[3], input_shape[4]
        oh, ow, od = h * self.size, w * self.size, d * self.size
        oc = c // (self.size * self.size *self.size)

        out = K.reshape(inputs, (-1, h, w, d, self.size, self.size, self.size, oc))
        out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 6, 5, 7))
        out = K.reshape(out, (-1, oh, ow, od, oc))
        return out

    def compute_output_shape(self, input_shape):
        h, w, d, c = input_shape[1], input_shape[2], input_shape[3], input_shape[4]
        oh, ow, od = h * self.size, w * self.size, d * self.size
        oc = c // (self.size * self.size *self.size)
        
        return input_shape[0], oh, ow, od, oc
