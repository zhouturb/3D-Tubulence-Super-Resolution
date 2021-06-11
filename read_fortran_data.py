# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import FortranFile
from concatenate import *

root = 'D:/_Isotropic_turbulence_/iso_tur_DNS1024_coarse/iso_1024_JHTDB/'
root_save = 'E:/TVSRCNN/data_0508/1024_JHTDB/FDNS_gauss_r4/'

sub_v = os.listdir(root)

all_u = []
all_v = []
all_w = []

# 512 data:
# for i in range(1):
#     for j in range(i*64, i*64+64):
#         f = FortranFile(root+sub_v[j],'r')
#         tmp_u = f.read_reals('f8')
#         tmp_v = f.read_reals('f8')
#         tmp_w = f.read_reals('f8')
#         all_u.append(tmp_u)
#         all_v.append(tmp_v)
#         all_w.append(tmp_w)
#     f.close()
#     Vu = concat_fortran_data_1024(all_u, sub=32, size=32)   # 128/8=sub, 256/8=sub。。。
#     Vv = concat_fortran_data_1024(all_v, sub=32, size=32)
#     Vw = concat_fortran_data_1024(all_w, sub=32, size=32)
#     plt.imshow(Vu[0],cmap=plt.cm.RdBu)
#     plt.show()

#     np.save(root_save+'u/u-'+str(i+1)+'.npy', Vu)
#     np.save(root_save+'v/v-'+str(i+1)+'.npy', Vv)
#     np.save(root_save+'w/w-'+str(i+1)+'.npy', Vw)

# 1024 data
for i in range(1):
    for j in range(len(sub_v)):
        f = FortranFile(root+sub_v[j],'r')
        tmp_u = f.read_reals('f8')
        tmp_v = f.read_reals('f8')
        tmp_w = f.read_reals('f8')
        print('shape=', tmp_u.shape)
        all_u.append(tmp_u)
        all_v.append(tmp_v)
        all_w.append(tmp_w)
    f.close()
    Vu = concat_fortran_data_1024(all_u, sub=16, size=16)   # 128/8=sub, 256/8=sub。。。
    Vv = concat_fortran_data_1024(all_v, sub=16, size=16)
    Vw = concat_fortran_data_1024(all_w, sub=16, size=16)
    plt.imshow(Vu[:,:,0],cmap=plt.cm.RdBu)
    plt.show()

    np.save(root_save+'u/u-'+str(i+1)+'.npy', Vu)
    np.save(root_save+'v/v-'+str(i+1)+'.npy', Vv)
    np.save(root_save+'w/w-'+str(i+1)+'.npy', Vw)
