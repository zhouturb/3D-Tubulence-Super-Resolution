# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.layers import Activation, concatenate
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from EDSR import generator
#from FFTSR import generator
# from RCAN import generator
# from SKDR import generator

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
devices  = ['/device:gpu:0', '/device:gpu:1']
strategy = tf.distribute.MirroredStrategy(devices=devices,cross_device_ops=tf.distribute.ReductionToOneDevice())
print('Number of devices: %d' % strategy.num_replicas_in_sync)
class TVSRCNN():
    def __init__(self, lr_size, model_path, from_direct=False, is_training=False, learning_rate=1e-4, batch_size=16, epochs=100):
        self.lr_size = lr_size    # 低分辨率输入尺寸
        self.path = model_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.from_direct = from_direct
        self.is_training = is_training
        if self.is_training:
            self.model = self.build_model()
        else:
            self.model = self.load()
        self.call_back_list =[
                ModelCheckpoint(filepath='./weight/'+self.path,
                                monitor='loss', save_best_only=True),
                ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5,
                                  verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)]
        
    #                EarlyStopping(monitor='val_loss', patience=5),
    
    def build_model(self):
        shape = (self.lr_size, self.lr_size, self.lr_size, 1)

        model = generator(input_shape=shape)
        model.summary()
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.MSE])
     
        return model
    

    def loss(self, y_true, y_pred):
        '''
        自定义loss, 每个速度变量的mse和合速度的mse
        '''
#        def _logcosh(x):
#            return x + K.softplus(-2.0 * x) - K.log(2.0)
#        loss1 = K.mean(_logcosh(y_pred - y_true), axis=-1)        
        
        def _sum_square(x):
            return K.sum(K.square(x), axis=-1)
        loss2 = K.mean(K.square(_sum_square(y_pred) - _sum_square(y_true)), axis=-1)
        
        return loss2
    
    
    def train(self, train_X, train_Y, test_x=0, test_y=0):
        
        if self.from_direct:
            train_generator = train_X
            history = self.model.fit_generator(train_generator, validation_data=[test_x,test_y],
                                               epochs=self.epochs, steps_per_epoch=5470)
        else:
            
            history = self.model.fit(train_X, train_Y, batch_size=self.batch_size, epochs=self.epochs, 
                                     verbose=1, callbacks=self.call_back_list, validation_split=0.1)
        if self.is_training:
            self.save()
        
        return history
    
        
    def process(self, input_X):
        predicted = self.model.predict(input_X)
        
        return predicted
    
    def load(self):
        weight_file = './weight/'+self.path
        model = self.build_model()
        model.load_weights(weight_file)
        
        return model
        
    def save(self):
        self.model.save_weights('./weight/'+self.path)
        
    
        
    
        
    
    
        

