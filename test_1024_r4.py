# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from model_single_gpu import TVSRCNN
import matplotlib.pyplot as plt
from concatenate import concat
from tqdm import tqdm
from split_data import data_patch_1024
from scipy.io import FortranFile



model = 'tvsr_weight_gauss_r4_0701.h5'

def main(test_u, test_v, test_w, test_y):  
    tvsrcnn = TVSRCNN(lr_size=32, model_path=model, is_training=False)
    tmp_u = []
    tmp_v = []
    tmp_w = []
    for i in tqdm(range(512)):
        tmp_u.append(tvsrcnn.process(test_u[i]))
        tmp_v.append(tvsrcnn.process(test_v[i]))
        tmp_w.append(tvsrcnn.process(test_w[i]))
    pred_u = concat(tmp_u, sub=128, size=8)
    pred_v = concat(tmp_v, sub=128, size=8)
    pred_w = concat(tmp_w, sub=128, size=8)
    plt.figure(figsize=(7,7))
    plt.imshow(pred_v[0], cmap=plt.cm.RdBu_r)
    plt.show()
    plt.figure(figsize=(7,7))
    plt.imshow(test_y[0,0,:,:,0], cmap=plt.cm.RdBu_r)
    
    # np.savetxt('./results/pred_u_1024_gauss_0706.txt', pred_u.reshape(-1))
    # np.savetxt('./results/pred_v_1024_gauss_0706.txt', pred_v.reshape(-1))
    # np.savetxt('./results/pred_w_1024_gauss_0706.txt', pred_w.reshape(-1))
    

    pred_u = data_patch_1024(pred_u, mesh_size=1024, box_size=64, dist=64)
    pred_v = data_patch_1024(pred_v, mesh_size=1024, box_size=64, dist=64)
    pred_w = data_patch_1024(pred_w, mesh_size=1024, box_size=64, dist=64)
    print(pred_u.dtype, pred_u.shape)


    idx = 0
    for i in range(16):
        for j in range(16):
            f = FortranFile('./results/JHTDB_1024_gauss_r4_pred/JHTDB_1024_gauss_r4_pred.%04d%04d'%(i,j), 'w')
            f.write_record(pred_u[idx])
            print('shape=', pred_u[idx].shape)
            f.write_record(pred_v[idx])
            f.write_record(pred_w[idx])
            f.close()
            idx += 1
              
if __name__ == '__main__':
    root = 'E:/TVSRCNN/dataset/'
    test_x = np.load(root+'test_x/test_x_1024_JHTDB_gauss_r4.npy')
    test_y = np.load(root+'test_y/test_y_512_gauss_t3.npy') 
    test_u = test_x[:,:,:,:,0].reshape(512,1,32,32,32,1)
    test_v = test_x[:,:,:,:,1].reshape(512,1,32,32,32,1)
    test_w = test_x[:,:,:,:,2].reshape(512,1,32,32,32,1)
    root_x = 'E:/TVSRCNN/data_0305/512/gauss_r4/FDNS/u/'
    load_Fu =  pd.read_table(root_x +'Flow_FDNS_u_3D10001.dat', header=None, sep='\s+').values
    test_fu = load_Fu.reshape(128,128,128)
    plt.figure(figsize=(7,7))
    plt.imshow(test_fu[0], cmap=plt.cm.RdBu_r)
    plt.show()
    main(test_u, test_v, test_w, test_y)