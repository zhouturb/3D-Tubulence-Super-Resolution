# -*- coding: utf-8 -*-

import os
import numpy as np

'制作train_dataset数据集'

root = 'E:/TVSRCNN/dataset/'
filter_type = '1024_r8_gauss'
x_files = os.listdir(root+'train_x/%s/'%(filter_type))
y_files = os.listdir(root+'train_y/%s/'%(filter_type))

print(x_files)
print(y_files)

temp_x = []
temp_y = []
for i in range(len(x_files)):
    
    x = np.load(root+'train_x/%s/'%(filter_type)+x_files[i])
    y = np.load(root+'train_y/%s/'%(filter_type)+y_files[i])
    temp_x.append(x)
    temp_y.append(y)
    
train_x = np.concatenate(temp_x, axis=0)
train_y = np.concatenate(temp_y, axis=0)

np.save(root+'train_x/%s/train_x_%s.npy'%(filter_type,filter_type), train_x)
np.save(root+'train_y/%s/train_y_%s.npy'%(filter_type,filter_type), train_y)
