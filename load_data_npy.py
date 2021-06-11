# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(x_data, mesh_size=64, box_size=16, dist=16):
    '''
    mesh_size: 计算域网格尺寸
    box_size:  切块域网格尺寸
    dist    :  采样间隔
    '''
    
    size = mesh_size
    data = x_data.reshape(size, size, size)    
    print(data.shape)
    train_x = []

    for i in range(0, size-box_size+1, dist):
        for j in range(0, size-box_size+1, dist):
            for k in range(0, size-box_size+1, dist):
                tmp = data[i:i+box_size, j:j+box_size, k:k+box_size]
                tmp = tmp.reshape(box_size,box_size,box_size,1)
                train_x.append(tmp)
    
    train_x = np.array(train_x)
                
    return train_x

def load_patch_data(x_data, mesh_size=64, box_size=16, dist=16):
    '''
    mesh_size: 计算域网格尺寸
    box_size:  切块域网格尺寸
    dist    :  采样间隔
    '''
    
    size = mesh_size
    temp = x_data.reshape(size, size, size)    
    
    # 处理边界问题
    s = mesh_size + dist
    m = np.int16(dist/2)
    data = np.zeros((s, s, s))
    print(data.shape)
    data[m:-m, m:-m, m:-m] = temp
    train_x = []

    for i in range(0, s-box_size+1, dist):
        for j in range(0, s-box_size+1, dist):
            for k in range(0, s-box_size+1, dist):
                tmp = data[i:i+box_size, j:j+box_size, k:k+box_size]
                tmp = tmp.reshape(box_size,box_size,box_size,1)
                train_x.append(tmp)
    
    train_x = np.array(train_x)
                
    return train_x
                
def data_patch(root, mesh_size=64, box_size=16, dist=16):
    u_name = os.listdir(root+'u')
    v_name = os.listdir(root+'v')
    w_name = os.listdir(root+'w')
    print(u_name)
    
    x = []
    for i in range(1):
         
        load_Fu = np.load(root+'u/'+u_name[i])    
        load_Fv = np.load(root+'v/'+v_name[i])   
        load_Fw = np.load(root+'w/'+w_name[i])
    
        test_Fu = load_data(load_Fu, mesh_size, box_size, dist)
        test_Fv = load_data(load_Fv, mesh_size, box_size, dist)
        test_Fw = load_data(load_Fw, mesh_size, box_size, dist)
        
        x.append(test_Fu)
        x.append(test_Fv)
        x.append(test_Fw)
    test_X = np.concatenate(x, axis=0)
 
    return test_X

def test_data_patch(root, mesh_size=128, box_size=8, dist=8):
    u_name = os.listdir(root+'u')
    v_name = os.listdir(root+'v')
    w_name = os.listdir(root+'w')
    
    x = []
    for i in range(1):
         
        load_Fu = np.load(root+'u/'+u_name[i])    
        load_Fv = np.load(root+'v/'+v_name[i])   
        load_Fw = np.load(root+'w/'+w_name[i])
    
        test_Fu = load_data(load_Fu, mesh_size, box_size, dist)
        test_Fv = load_data(load_Fv, mesh_size, box_size, dist)
        test_Fw = load_data(load_Fw, mesh_size, box_size, dist)
        
        test_x = np.concatenate([test_Fu, test_Fv, test_Fw], axis=-1)
        x.append(test_x)

    test_X = np.concatenate(x, axis=0)
 
    return test_X

def train_dataset(root):
    train_X = data_patch(root+'FDNS_r8/', mesh_size=128,  box_size=16,  dist=16)
    train_Y = data_patch(root+'DNS/',     mesh_size=1024, box_size=128, dist=128)
    
    return train_X, train_Y

def test_dataset(root):
    test_X = test_data_patch(root, mesh_size=256, box_size=32, dist=32)
    # test_Y = test_data_patch(root+'DNS/', mesh_size=512, box_size=512, dist=512)
    test_Y = 0
    
    return test_X, test_Y

def les_dataset(root):
    test_X = test_data_patch(root+'FDNS/', mesh_size=64, box_size=16, dist=8)
    
    return test_X

    
root = 'E:/TVSRCNN/data_0508/1024/'
train_x, train_y = train_dataset(root)

root_save = 'E:/TVSRCNN/dataset/'
np.save(root_save+'train_x/1024_r8_gauss/train_x_1024_r8_gauss.npy', train_x)
np.save(root_save+'train_y/1024_r8_gauss/train_y_1024_r8_gauss.npy', train_y)
# np.save(root_save+'test_x/test_x_1024_JHTDB_gauss_r4.npy',train_x)
# np.save(root_save+'test_y/test_y_512_gauss_r8_t3.npy',train_y)
