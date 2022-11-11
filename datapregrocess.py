#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 15:43:28 2022

@author: hbr
"""

import os 
import pickle 
import numpy as np 
import tensorflow as tf 


def CMS(file='./Mu_Sig_C_Seg1_120km.pkl'):

    with open(file, 'rb') as f:
        dic=pickle.load(f)

    C = dic['C'][:,5:,:]
    C = np.transpose(C,(0,2,1))
    C = np.reshape(C,(18,300))
    
    mu = dic['Mu'][:,5:,:]
    mu = np.transpose(mu, (0,2,1))
    mu = np.reshape(C, (18,300))
    
    std = dic['Sigma'][:,5:,:]
    std = np.transpose(std, (0,2,1))
    std = np.reshape(std, (18,300))

    return mu.astype('float32'), std.astype('float32'), C.astype('float32')


def loadCBA(file='./Mu_Sig_C_Seg1_120km.pkl'):

    with open(file, 'rb') as f:
        dic=pickle.load(f)

    C = dic['C'][:,5:,:]
    C = np.transpose(C,(0,2,1))
    C = np.reshape(C,(18,300))
    
    B = dic['B'][:,5:,:]
    B = np.transpose(B,(0,2,1))
    B = np.reshape(B,(18,300))
    
    A = dic['A'][:,5:,:]
    A = np.transpose(A,(0,2,1))
    A = np.reshape(A,(18,300))

    return A.astype('float32'), B.astype('float32'), C.astype('float32')


"""pt, lat, lon, R, tmat"""
class datagress():
    
    def __init__(self, IN=9., Type='all', NormTy='Norm0', file='./Mu_Sig_C_Seg1_120km.pkl'):
        
        super(datagress, self).__init__()
        self.mu, self.std, self.C = CMS(file)
        self.Type = Type
        self.NormTy = NormTy 
        self.IN = np.float32(IN) 
        
        
    def Norm0(self, x):
        
        x = x - self.mu[np.newaxis,np.newaxis,:,:,np.newaxis]
        x = x / self.std[np.newaxis,np.newaxis,:,:,np.newaxis]
        
        return x 
    
    def Norm1(self, x):
        
        x = x / self.max[np.newaxis,np.newaxis,:,:,np.newaxis]
        
        return x 
    
    def __call__(self, x0, x1, x2, y):
        
        if self.NormTy=='Norm0':
            R = np.log10(x2[:,:,-2] + 1e-6)
            R = R[:,:,np.newaxis,np.newaxis,np.newaxis]
            x0 = x0 + self.C[np.newaxis,np.newaxis,:,:,np.newaxis] * (1.-R)
            x0 = self.Norm0(x0)
            
        if  self.NormTy=='Norm1':
            x0 = self.Norm1(x0)
            
        x0 = x0 * x1[:,:,np.newaxis,:,np.newaxis]
        if self.Type=='T1':
            x0 = x0[:,:,:9,:,:]
        elif self.Type=='T2':
            x0 = x0[:,:,:12,:,:]
        elif self.Type=='T3':
            x0 = x0[:,:,:15,:,:]
        
        num = x0.shape[2]//3
        t = np.ones((1,1,num)) * np.max(x1, axis=-1, keepdims=True) 
        s0, s1, s2 = t.shape
        t = np.reshape(t, (s0, 1, -1))
        t = t.astype('float32')
        
        y = y /self.IN 
        
        return x0, t, y
    

class MagCallinr():
    
    def __init__(self, Type='all', file='./Mu_Sig_C_Seg1_120km.pkl'):
        
        super(MagCallinr, self).__init__()
        self.A, self.B, self.C = loadCBA(file)
        self.Type = Type
        
    def __call__(self, x0, x1, x2, y):
        
        R = np.log10(x2[:,:,-2] + 1e-6)
        R = R[:,:,np.newaxis,np.newaxis,np.newaxis]
        
        C = self.C[np.newaxis,np.newaxis,:,:,np.newaxis]
        B = self.B[np.newaxis,np.newaxis,:,:,np.newaxis]
        A = self.A[np.newaxis,np.newaxis,:,:,np.newaxis]
        
        PM = (x0 - C * R - A) / B
            
        PM = PM * x1[:,:,np.newaxis,:,np.newaxis]
        if self.Type=='T1':
            PM = PM[:,:,:9,:,:]
        elif self.Type=='T2':
            PM = PM[:,:,:12,:,:]
        elif self.Type=='T3':
            PM = PM[:,:,:15,:,:]
        
        num = PM.shape[2]//3
        t = np.ones((1,1,num)) * np.max(x1, axis=-1, keepdims=True) 
        s0, s1, s2 = t.shape
        t = np.reshape(t, (s0, 1, -1))
        t = t.astype('float32')
                
        return PM.astype('float32'), t