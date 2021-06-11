# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from model_single_gpu import TVSRCNN
# from model import TVSRCNN

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#------------------------load_data-------------------------------#
filter_type = '1024_r8_gauss'
# filter_type = 'gauss'
root = 'E:/TVSRCNN/dataset/'
train_x = root+'train_x/%s/train_x_%s.npy'%(filter_type, filter_type)
train_y = root+'train_y/%s/train_y_%s.npy'%(filter_type, filter_type)
# train_x = root+'train_x/channel/train_x.npy'
# train_y = root+'train_y/channel/train_y.npy'

model = 'tvsr_weight_1024_gauss_r8_210201.h5'


def main():
    
    train_X = np.load(train_x)
    train_Y = np.load(train_y)
    print(train_X.shape)
    print(train_Y.shape)
    
    # data shuffle:
    index = [i for i in range(len(train_X))]
    np.random.shuffle(index)
    train_X = train_X[index]
    train_Y = train_Y[index]
    
    tvsrcnn = TVSRCNN(lr_size=16, model_path=model, is_training=True, learning_rate=1e-4, batch_size=3, epochs=150)
    history = tvsrcnn.train(train_X, train_Y)
    df_results = pd.DataFrame(history.history)
    df_results['epoch'] = history.epoch
    df_results.to_csv(path_or_buf='./history/History_tvsr_1024_gauss_r8_210201.csv',index=False)
    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    main()
    
    
    
    
