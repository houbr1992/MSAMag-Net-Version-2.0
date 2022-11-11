#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 09:35:56 2022

@author: hbr
"""


import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import ReLU, LeakyReLU, MultiHeadAttention,  LayerNormalization, GlobalMaxPool1D


class sinparnet(tf.keras.layers.Layer):

    def __init__(self,
                 filters=[2,4,8,16,32,64],
                 kr_sz=[(3,3), 3],
                 ss_sz=[(3,2), 2],
                 alp = 1.,
                 inp_shp0=[0,1,2,3,4],
                 inp_shp1=[0,1,2,3]):
        super(sinparnet, self).__init__()

        self.cnn_bk0 = [ Conv2D(filters[0], kr_sz[0], ss_sz[0], padding='same', 
                                kernel_initializer='glorot_uniform',
                                bias_initializer='zeros',
                                input_shape=inp_shp0[1:])
                         for i in range(1) ]

        self.cnn_bk1 = [ Conv1D(filters[i], kr_sz[1], ss_sz[1], padding='same', 
                                kernel_initializer='glorot_uniform',
                                bias_initializer='zeros', 
                                input_shape=inp_shp1[1:])
                         for i in range(1,len(filters))]

        self.MaxP = MaxPool2D(pool_size=(1, 2), padding='same')
        self.act = tf.keras.activations.elu
        self.alp = alp 

    def __call__(self, x):
        
        laydic = {}
        num = 1 
        for CNN in self.cnn_bk0:
            x = CNN(x)
            laydic[f'lay_{num}_CNN'] = x 
            x = self.act(x, self.alp)
            laydic[f'lay_{num}_Act'] = x 
            num += 1 
        x = tf.squeeze(x, axis=2)
        
        for CNN in self.cnn_bk1:
            x = CNN(x)
            laydic[f'lay_{num}_CNN'] = x 
            x = self.act(x, self.alp)
            laydic[f'lay_{num}_Act'] = x 
            x = self.MaxP(x)
            laydic[f'lay_{num}_MaxP'] = x 
            num += 1 

        return x, laydic 
    
    
class FFNet(tf.keras.layers.Layer):
    """
    x -> (None, -1, dim)
    x -> (None, -1, dim)
    dim -> 32 
    dff -> 64
    """
    def __init__(self, dim=32, dff=64, alp=1., droprate=0.3):
        super(FFNet, self).__init__()
        self.Den0 = Dense(dff, None, True, 'glorot_uniform', 'zeros')
        self.Den1 = Dense(dim, None, True, 'glorot_uniform', 'zeros')
        self.Drop = Dropout(droprate)
        self.act = tf.keras.activations.elu
        self.alp = alp
        
    def __call__(self, x, TS):
        
        x = self.Den0(x)
        x = self.Drop(x, TS)
        x = self.act(x, self.alp)
        x = self.Den1(x)
        
        return x 
    

class EnLay(tf.keras.layers.Layer):
    """
    Inp: x -> (None, 4, dim)
    Out: x -> (None, 4, dim)
    dff: 64
    head: 4 
    kdim: 8 dim // heads np.sqrt(dk)
    vdim: 8 dim // heads
    """
    def __init__(self, dff=64, kdim=8, vdim=8, dim=32, heads=4, alp=1., droprate=0.3):
        super(EnLay, self).__init__()
        self.mha = MultiHeadAttention(num_heads=heads, key_dim=kdim, value_dim=vdim)
        self.ffn = FFNet(dim, dff, alp, droprate)
        self.lay1 = LayerNormalization(epsilon=1e-6)
        self.lay2 = LayerNormalization(epsilon=1e-6)
    
    def __call__(self, x, mask=None, TS=False):
        
        x1 = self.lay1(x)
        att_out, wei = self.mha(x1, x1, x1, attention_mask=mask, 
                                        return_attention_scores=True) 
        att_out = tf.math.add(att_out, x)
        
        att_out1 = self.lay2(att_out)                                                  
        ffn_out = self.ffn(att_out1, TS)
        ffn_out = tf.math.add(ffn_out, att_out)     
        
        return ffn_out, wei 
    
    
class EnNet(tf.keras.layers.Layer):
    """
    Encoder Net
    Inp: x -> None, -1, dim 
    Out: x -> None, -1, dim 
    """
    def __init__(self, numly0=3, dff=64, kdim=8, vdim=8, dim=32, heads=4, alp=1., droprate=0.3):
        super(EnNet, self).__init__()
        self.EN_BK = [EnLay(dff, kdim, vdim, dim, heads, alp, droprate) for i in range(numly0)]
        
    def __call__(self, x, mask=None, TS=False):
        
        att_wei = {}
        laydic = {}
        i = 0 
        laydic[f'enc_lay_{i}'] = x.numpy().astype('float32')  
        for Att in self.EN_BK:
            x, wei = Att(x, mask, TS)
            att_wei[f'enc_lay_{i+1}_block'] = wei.numpy().astype('float32')  
            mask1 =  tf.transpose(mask, [0,2,1])
            laydic[f'enc_lay_{i+1}'] = (x * mask1).numpy().astype('float32') 
            i = i + 1 
            
        return x, att_wei, laydic 
    
    
class MSAMag1(tf.keras.layers.Layer):

    def __init__(self,
                 filters0=[2,4,8,16,32,64],
                 kr_sz0=[(3,3), 3],
                 ss_sz0=[(3,2), 2],
                 numly0=3, dff=64,
                 kdim=8, vdim=8, dim=32, 
                 heads=4, alp=1., droprate=0.3, 
                 dimL=1):
        super(MSAMag1, self).__init__()
        
        self.CNN0 = sinparnet(filters0, kr_sz0, ss_sz0, alp)
        self.CNN1 = sinparnet(filters0, kr_sz0, ss_sz0, alp)
        self.CNN2 = sinparnet(filters0, kr_sz0, ss_sz0, alp)
        
        self.ENet = EnNet(numly0, dff, kdim, vdim, dim, heads, alp, droprate)
        self.GMP = GlobalMaxPool1D()
        self.LNet = Dense(dimL)
        
    def __call__(self, x, mask=None, TS=False):
        
        x0, x1, x2 = x[:,:,:3,:,:], x[:,:,3:6,:,:], x[:,:,6:9,:,:]
        
        x0, laydic0 = self.CNN0(x0) 
        x1, laydic1 = self.CNN1(x1) 
        x2, laydic2 = self.CNN2(x2) 
        
        x = tf.concat([x0, x1, x2], axis=2)
        
        s0, s1, s2, s3 = x.shape
        x = tf.reshape(x, (s0, -1, s3))
        
        #Attention Net 
        x, att_wei, laydic3 = self.ENet(x, mask, TS)        
        mask = tf.transpose(mask, [0,2,1])
        x = mask * x
        x = self.GMP(x)
        
        laydic4 = {}
        laydic4['Att_MaxP'] = x.numpy().astype('float32')  
        
        #Last Net
        x = self.LNet(x)
        laydic = {}
        laydic['Pa'] = laydic0 
        laydic['Pv'] = laydic1 
        laydic['Pd'] = laydic2 
        laydic['encoders'] = laydic3 
        laydic['encoders_MaxP'] = laydic4         

        return x, att_wei, laydic 
    

class ILoss(tf.keras.layers.Layer):
    """
    2 Losses: MSE, MAE
    """
    def __init__(self,Type='MSE'):
        super(ILoss, self).__init__()
        self.Type = Type
        self.MSE = tf.keras.losses.MeanSquaredError()
        self.MAE = tf.keras.losses.MeanAbsoluteError()
        
    def __call__(self, y_true, y_pred):
        
        y_true = y_true[:, tf.newaxis]
        
        if self.Type == 'MSE':
            loss = self.MSE(y_true, y_pred)
            
        if self.Type == 'MAE':
            loss = self.MAE(y_true, y_pred)
            
        return loss 