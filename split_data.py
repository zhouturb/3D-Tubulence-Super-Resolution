# -*- coding: utf-8 -*-

import os
import numpy as np

def load_data(x_data, mesh_size=64, box_size=16, dist=16):
    '''
    mesh_size: 计算域网格尺寸
    box_size:  切块域网格尺寸
    dist    :  采样间隔
    '''
    
    size = mesh_size
    data = x_data.reshape(size, size, size)    
    train_x = []

    for i in range(0, size-box_size+1, dist):
        for j in range(0, size-box_size+1, dist):
                tmp = data[j:j+box_size,i:i+box_size]
#                tmp = tmp.swapaxes(0,2)
#                tmp = tmp.swapaxes(1,2)
                train_x.append(tmp)   
    train_x = np.array(train_x, dtype=np.float64)                
    return train_x


def data_patch(inputs, mesh_size=64, box_size=16, dist=16):

    for i in range(1):       
        load_Fu =  inputs
        test_Fu = load_data(load_Fu, mesh_size, box_size, dist)

    return test_Fu


def load_data_1024(x_data, mesh_size=64, box_size=16, dist=16):
    '''
    mesh_size: 计算域网格尺寸
    box_size:  切块域网格尺寸
    dist    :  采样间隔
    '''
    
    size = mesh_size
    data = x_data.reshape(size, size, size)    
    train_x = []

    for i in range(0, size-box_size+1, dist*2):
        for j in range(0, size-box_size*2+1, dist*4):
                tmp = data[i:i+box_size,j:j+box_size*2]
#                tmp = tmp.swapaxes(0,2)
#                tmp = tmp.swapaxes(1,2)
                train_x.append(tmp)   
    train_x = np.array(train_x, dtype=np.float64)                
    return train_x


def data_patch_1024(inputs, mesh_size=64, box_size=16, dist=16):

    for i in range(1):       
        load_Fu =  inputs
        test_Fu = load_data(load_Fu, mesh_size, box_size, dist)

    return test_Fu


# a = np.ones((1024,1024,1024))
a = np.ones((512,512,512))
# pred_u = data_patch_1024(a, mesh_size=1024, box_size=32, dist=16)
# pred_v = data_patch_1024(a, mesh_size=1024, box_size=32, dist=16)
pred_w = data_patch_1024(a, mesh_size=512, box_size=64, dist=64)
