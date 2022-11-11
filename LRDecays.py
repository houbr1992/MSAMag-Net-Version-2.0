# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 10:11:30 2022

@author: hbrhu
"""
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) 

# dmodel = 64
# warmup_steps = 512
# temp_learning_rate_schedule = CustomSchedule(dmodel, warmup_steps)

# plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
# plt.ylabel('Learning Rate')
# plt.xlabel('Train Step')


class CustSchdle(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, init_lr, lr_min=1e-5, warmup_steps=2048):
    super(CustSchdle, self).__init__()
    
    self.init_lr = tf.cast(init_lr, tf.float32) 
    self.warmup_steps = warmup_steps
    self.lr_min = tf.cast(lr_min, tf.float32)

  def __call__(self, step):
      if step < self.warmup_steps:
          warmup_percent_done = step / self.warmup_steps
          warmup_lr = self.init_lr * warmup_percent_done
          lr = warmup_lr
      else:
          num_init = step// self.warmup_steps
          lr = self.init_lr * 0.9**(1.2*num_init)
          lr = tf.math.maximum(lr, self.lr_min)
      return lr
  
# warmup_steps = 256
# init_lr = 0.01
# lr_min = 1e-5
# temp_learning_rate_schedule = CustSchdle(init_lr, lr_min, warmup_steps)
# ylis = []
# xlis = []
# for step in range(40000):
#     ylis.append(temp_learning_rate_schedule(step))
#     xlis.append(step)

# y1 = np.array(ylis)
# x1 = np.array(xlis)
# plt.plot(x1,y1)
# plt.ylabel('Learning Rate')
# plt.xlabel('Train Step')