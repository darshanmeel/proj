# -*- coding: utf-8 -*-
"""
Created on Sat Jul 04 13:26:22 2015

@author: Inpiron
"""
from autoencoder_single_hidden_layer import autoencoder_single_hidden_layer
from NN_single_hidden_layer import NN_single_hidden_layer
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from active_functions import *
from loss_functions import *

iris = datasets.load_iris()
dt = np.array(iris.data)
lbl = np.array(iris.target)#.reshape(dt.shape[0],1)

print dt.shape
print lbl.shape

train_data,test_data,train_class,test_class = train_test_split(dt,lbl,train_size = 0.75,random_state=190876)

n_input = train_data.shape[1]
n_hidden = 2
n_out = len(np.unique(lbl))
n_out = n_input
print n_out
nn1 = autoencoder_single_hidden_layer(n_hidden,n_input,n_out,epochs = 200,batchsize=1,learning_rate=0.01,loss_fnc=rms_reg,reg = 0.0001,momentum = 0.95)
train_error,valid_error = nn1.fit(train_data,test_data)
    
print train_error
print
print valid_error

_,train_data = nn1.predict(train_data)
_,test_data = nn1.predict(test_data)

train_data = np.array(train_data)
test_data = np.array(test_data)
print
print 
n_input = train_data.shape[1]
n_hidden = 10
n_out = len(np.unique(lbl))

print n_out
nn1 = NN_single_hidden_layer(n_hidden,n_input,n_out,epochs = 1000,batchsize=1,learning_rate=0.1,loss_fnc=rms_cls,reg = 0.001,momentum = 0.95)
train_error,valid_error = nn1.fit(train_data,train_class,test_data,test_class)
    
print train_error
print
print valid_error

'''
print
print nn1.nn['w1']
print
print nn1.nn['b1']
print
print nn1.nn['w2']
print
print nn1.nn['b2']
'''