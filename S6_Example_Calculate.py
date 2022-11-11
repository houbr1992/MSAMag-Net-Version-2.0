#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 22:24:39 2022

@author: hbr
"""

import gc 
import os
import time
import pickle
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import chain
from tqdm import tqdm
from dataparamsline import dataparams
from datapregrocess import datagress, MagCallinr
from tfresfunc import tvtkdata, tvtkque
from model_layers import MSAMag1, ILoss


def ExamDic(Tlis, Type, ckth, dat, dat_ty, dirs):
    
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    
    dic = {}
    len_t = 30
    for f0 in tqdm(Tlis):
        
        indl = [32, 32, 32, 32, 32, 32, 32, 32]
        ratl = [12, 4, 3, 1, 1, 1, 1, 1]
        magl = [2.99, 3.49, 3.99, 4.49, 4.99, 5.49, 5.99, 6.49]
        Data1 = dataparams(indl, ratl, magl)
        
        # Sigle Event
        eve_dic = {}
        f0 = [f0]
        IO = [[] for i in range(8)]

        for t0 in range(1, 31, 1):
            IO_ = Data1(f0, t0)
            IO = [IO[i] + IO_[i] for i in range(8)]

        Qlat, Qlon, depth, event = IO[4][0], IO[5][0], IO[6][0], IO[7][0]
        IO = [np.array(IO[i]) for i in range(4)]
        x0, x1, x2, x3 = IO
        s0, s1, s2, s3, s4 = x0.shape
        x0 = np.reshape(x0, (s0, s1, -1, s4, 1))
        PML, tlinr = MagClinr(x0, x1, x2, x3)
        x0, mask, y = data(x0, x1, x2, x3)
        # print(x0.shape)
        # print(mask.shape)
        
        out, wei, lay = model(x0, mask)
        out = out.numpy()
        if s0 < len_t:
            s = len_t - s0
            out = np.pad(out, ((s, 0), (0, 0)), constant_values=(-1, 0))
            x1 = np.pad(x1, ((s, 0), (0, 0), (0, 0)), constant_values=(-1, 0))
        trinum = np.sum(np.greater_equal(np.max(x1, axis=-1), 1) + 0., axis=-1)
        t30 = x1[-1].astype('float32')
        
        evedic = {'event':event, 'Mag':x3[0], 'Depth':depth,
                  'Qlat':Qlat, 'Qlon':Qlon,  
                  'PreM':out*9., 'PML': PML[-1,:,:,:,0],
                  'TriN':trinum, 't30':t30, 
                  'AttWei':wei, 'layer':lay}        
        dic[event] = evedic
        
    filsav = f'{dirs}/examples_{Type}_{dat}{dat_ty}_{ckth}.pkl'
    with open(filsav, 'wb') as f:
        pickle.dump(dic, f)
    
    return True 

def Exams_list(file, dirf, name='_Params_Type1'):
    
    with open(file, 'rb') as f:
        dic = pickle.load(f)
    
    Examslist = [os.sep.join([dirf, k[0:4], k+name]) for k in dic.keys()]
    
    return Examslist 

# Data Piplelines
dat_ty_lis = [[[d, t] for t in ['']] for d in ['snr2y']]
dat_ty_lis = list(chain(*dat_ty_lis))

Type = 'T1'
Rdat = '200km'
Tralis, Vallis, Teslis, Kumlis = [], [], [], []

for dat, dat_ty in dat_ty_lis:
            
    # TVTK Event list
    dirin = '/media/hbr/machine/Type1'
    dirsave = './'
    if dat == 'snr2y':
        dirf =  f'{dirin}/ParamsSNR2/event'
        
    indl = [32, 32, 32, 32, 32, 32, 32, 32]
    ratl = [12, 4, 3, 1, 1, 1, 1, 1]
    magl = [2.99, 3.49, 3.99, 4.49, 4.99, 5.49, 5.99, 6.49]
    Data1 = dataparams(indl, ratl, magl)
    
    Magf = 'pkl/Mag_TVT_4Reg2E_rmLSNR.pkl' 
        
            
    TraLis, ValLis, TesLis, KumLis = tvtkque(Magf, dirf)
    Tralis = list(chain(*TraLis))
    Vallis = list(chain(*ValLis))
    Teslis = list(chain(*TesLis))
    Kumlis = list(chain(*KumLis))
    
    # model initializer
    filters0=[2,4,8,16,32]
    kr_sz0=[(3,3), 3]
    ss_sz0=[(3,2), 2]
    dff=64
    dim=32 
    kdim=dim
    vdim=dim
    heads=1
    alp=1. 
    droprate=0.3 
    dimL=1
    numly0 = 6 
    num = 512
    
    # Data Progressing
    IN = 9.             
    NormTy = 'Norm0'
     
    file = f'./pkl/Mu_Sigma_snr2_{Rdat}{dat_ty}.pkl'  
    fileCBA = f'./pkl/Mu_Sigma_snr2_{Rdat}{dat_ty}_ABC.pkl'  
        
    file_exams = 'pkl/Examples_1015.pkl'        
    data = datagress(IN, Type, NormTy, file) 
    MagClinr =  MagCallinr(Type, fileCBA)
    
    if Type == 'T1':
        model = MSAMag1(filters0, kr_sz0, ss_sz0, numly0, dff, kdim, vdim, dim, heads, alp, droprate, dimL)
    
    optimizer = tf.keras.optimizers.Adam(0.001)
    checkname = 'checkp.ckpt'
    checkpoint_path = f'./model_files/model_{Type}_{numly0}_{dat}{dat_ty}'
    max2keep = 200
    ckpt = tf.train.Checkpoint(model = model,optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, 
                                              checkpoint_name=checkname, max_to_keep=max2keep)
    cklis = [96]
                
    for ckth in cklis:
        
        check_name='./{}/{}-{}'.format(checkpoint_path, checkname, ckth)
        ckpt.restore(check_name)
        dirs = f'{dirsave}/figs&files/AW{dat_ty}/{Type}_{dat}{dat_ty}/{Type}_{ckth}_{dat}{dat_ty}'
        
        Tlis = Exams_list(file_exams, dirf)
        ExamDic(Tlis, Type, ckth, dat, dat_ty, dirs)
        