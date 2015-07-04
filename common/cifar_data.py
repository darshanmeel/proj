# -*- coding: utf-8 -*-
"""
Created on Sat Jul 04 18:29:37 2015

@author: Inpiron
"""

import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
from sklearn.cross_validation import train_test_split
import pandas as pd

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte
  
def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

# Load the raw CIFAR-10 data

cifar10_dir = './cifar-10-batches-py'
X_train, y_train, X_test, test_class = load_CIFAR10(cifar10_dir)
    
#split the train data inot train and valid. Keep valid as 15% of the data

print X_train.shape

train_data = X_train.reshape(X_train.shape[0],-1)

train_data,valid_data,train_class,valid_class = train_test_split(train_data,y_train,test_size = 0.15)

#now save the data to csv file



test_data = X_test.reshape(X_test.shape[0],-1)

train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)
valid_data = pd.DataFrame(valid_data)

train_class = pd.DataFrame(train_class)
test_class = pd.DataFrame(test_class)
valid_class = pd.DataFrame(valid_class)


valid_class.to_csv('./data/valid_class.csv',index=False)
test_class.to_csv('./data/test_class.csv',index=False)
train_class.to_csv('./data/train_class.csv',index=False)

valid_data.to_csv('./data/valid_data.csv',index=False)
test_data.to_csv('./data/test_data.csv',index=False)
train_data.to_csv('./data/train_data.csv',index=False)


