# -*- coding: utf-8 -*-

import numpy as np

#将预测的256FDNS数据拼接（最大只能预测128**3的数据，所以需要拼接）

def concat_256(data, sub=128, size=2):
    
    temp=0
    tmp_i = []
    for i in range(size):
        tmp_j = []
        for j in range(size):
            tmp_k = []
            for k in range(size):
                pred = data[temp].reshape(sub,sub,sub)
                tmp_k.append(pred)
                temp += 1
            pred = np.concatenate([tmp_k[0], tmp_k[1]], axis=2)
            tmp_j.append(pred)
        pred = np.concatenate([tmp_j[0], tmp_j[1]], axis=1)
        tmp_i.append(pred)
    
    result = np.concatenate([tmp_i[0], tmp_i[1]], axis=0)   
    return result

def concat_512(data, sub=128, size=4):
    
    temp=0
    tmp_i = []
    for i in range(size):
        tmp_j = []
        for j in range(size):
            tmp_k = []
            for k in range(size):
                pred = data[temp].reshape(sub,sub,sub)
                tmp_k.append(pred)
                temp += 1
            pred = np.concatenate([tmp_k[0], tmp_k[1], tmp_k[2], tmp_k[3]], axis=2)
            tmp_j.append(pred)
        pred = np.concatenate([tmp_j[0], tmp_j[1], tmp_j[2], tmp_j[3]], axis=1)
        tmp_i.append(pred)
    
    result = np.concatenate([tmp_i[0], tmp_i[1], tmp_i[2], tmp_i[3]], axis=0)   
    return result

def concat(data, sub=64, size=16):
    
    temp=0
    tmp_i = []
    for i in range(size):
        tmp_j = []
        for j in range(size):
            tmp_k = []
            for k in range(size):
                pred = data[temp].reshape(sub,sub,sub)
                tmp_k.append(pred)
                temp += 1
            pred = np.concatenate([tmp_k[i] for i in range(size)], axis=2)
            tmp_j.append(pred)
        pred = np.concatenate([tmp_j[i] for i in range(size)], axis=1)
        tmp_i.append(pred)
    
    result = np.concatenate([tmp_i[i] for i in range(size)], axis=0)
    return result

def concat_patch(data, mesh=128, patch=32, size=8):
    
    temp=0
    tmp_i = []
    for i in range(size):
        tmp_j = []
        for j in range(size):
            tmp_k = []
            for k in range(size):
                pred = data[temp].reshape(mesh,mesh,mesh)
                # 处理重叠部分
                pred = pred[patch:-patch, patch:-patch, patch:-patch]
                tmp_k.append(pred)
                temp += 1
            pred = np.concatenate([tmp_k[i] for i in range(size)], axis=2)
            tmp_j.append(pred)
        pred = np.concatenate([tmp_j[i] for i in range(size)], axis=1)
        tmp_i.append(pred)
    
    result = np.concatenate([tmp_i[i] for i in range(size)], axis=0)   
    return result

def concat_fortran_data(data, sub=16, size=8):
    
    temp=0
    tmp_i = []
    for i in range(1):
        tmp_j = []
        for j in range(size):
            tmp_k = []
            for k in range(size):
                pred = data[temp].reshape(sub,sub,sub*8)
                tmp_k.append(pred)
                temp += 1
            pred = np.concatenate([tmp_k[i] for i in range(size)], axis=0)
            tmp_j.append(pred)
        pred = np.concatenate([tmp_j[i] for i in range(size)], axis=1)
#        print(pred.shape)

    result = pred
    return 

def concat_fortran_data_1024(data, sub=16, size=16):
    
    temp=0
    tmp_i = []
    for i in range(1):
        tmp_j = []
        for j in range(size):
            tmp_k = []
            for k in range(size):
                pred = data[temp].reshape(sub,sub,sub*16)
                tmp_k.append(pred)
                temp += 1
            pred = np.concatenate([tmp_k[i] for i in range(size)], axis=0)
            # print(pred.shape)
            tmp_j.append(pred)
        pred = np.concatenate([tmp_j[i] for i in range(size)], axis=1)
        # print(pred.shape)
        
    result = pred
    return result
        
def concat_fortran_data_128(data, sub=16, size=8):
    
    temp=0
    tmp_i = []
    for i in range(1):
        tmp_j = []
        for j in range(size):
            tmp_k = []
            for k in range(size):
                pred = data[temp].reshape(sub,sub,sub*8)
                tmp_k.append(pred)
                temp += 1
            pred = np.concatenate([tmp_k[i] for i in range(size)], axis=0)
            # print(pred.shape)
            tmp_j.append(pred)
        pred = np.concatenate([tmp_j[i] for i in range(size)], axis=1)
        # print(pred.shape)

    result = pred
    return result