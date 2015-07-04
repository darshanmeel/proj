# -*- coding: utf-8 -*-
"""
Created on Sat Jul 04 18:51:32 2015

@author: Inpiron
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import RandomizedPCA as PCA
from sklearn.preprocessing import MinMaxScaler as MMS

# Here I will whiten the images and I will use a min max scaer so that all the values are beteen 0 and 1


def rgb2gray(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.144]) 
    
# read the data 
train_data = pd.read_csv('./data/train_data.csv')
train_data = np.array(train_data)
train_data = train_data.reshape(train_data.shape[0],32,32,3)

train_data_r = train_data[:,:,:,0].reshape(train_data.shape[0],-1)
train_data_g = train_data[:,:,:,1].reshape(train_data.shape[0],-1)
train_data_b = train_data[:,:,:,2].reshape(train_data.shape[0],-1)
train_data = None
#Min max  the data
ss_r = MMS(copy=False)
ss_r.fit(train_data_r)
ss_r.transform(train_data_r)

'''
#whiten the data
pc_r = PCA(n_components=None,whiten=True)
pc_r.fit(train_data_r)
train_data_r = pc_r.transform(train_data_r)
'''


#Min max  the data
ss_g = MMS(copy=False)
ss_g.fit(train_data_g)
ss_g.transform(train_data_g)

'''
#whiten the data
pc_g = PCA(n_components=None,whiten=True)
pc_g.fit(train_data_g)
train_data_g = pc_g.transform(train_data_g)
'''


#Min max  the data
ss_b = MMS(copy=False)
ss_b.fit(train_data_b)
ss_b.transform(train_data_b)

'''
#whiten the data
pc_b = PCA(n_components=None,whiten=True)
pc_b.fit(train_data_b)
train_data_b = pc_b.transform(train_data_b)
'''

#now transform the test and valid data using above model.
test_data = pd.read_csv('./data/test_data.csv')
test_data = np.array(test_data)

test_data = test_data.reshape(test_data.shape[0],32,32,3)

test_data_r = test_data[:,:,:,0].reshape(test_data.shape[0],-1)
test_data_g = test_data[:,:,:,1].reshape(test_data.shape[0],-1)
test_data_b = test_data[:,:,:,2].reshape(test_data.shape[0],-1)
test_data = None

ss_r.transform(test_data_r)
#test_data_r = pc_r.transform(test_data_r)

ss_g.transform(test_data_g)
#test_data_g = pc_g.transform(test_data_g)

ss_b.transform(test_data_b)
#test_data_b = pc_b.transform(test_data_b)

valid_data = pd.read_csv('./data/valid_data.csv')

valid_data = np.array(valid_data)

valid_data = valid_data.reshape(valid_data.shape[0],32,32,3)

valid_data_r = valid_data[:,:,:,0].reshape(valid_data.shape[0],-1)
valid_data_g = valid_data[:,:,:,1].reshape(valid_data.shape[0],-1)
valid_data_b = valid_data[:,:,:,2].reshape(valid_data.shape[0],-1)
valid_data = None

ss_r.transform(valid_data_r)
#valid_data_r = pc_r.transform(valid_data_r)

ss_g.transform(valid_data_g)
#valid_data_g = pc_g.transform(valid_data_g)

ss_b.transform(valid_data_b)
#valid_data_b = pc_b.transform(valid_data_b)

#now write this data to preprocessed folder

train_data = np.dstack((train_data_r,train_data_g,train_data_b)).reshape(train_data_r.shape[0],-1)
test_data = np.dstack((test_data_r,test_data_g,test_data_b)).reshape(test_data_r.shape[0],-1)
valid_data = np.dstack((valid_data_r,valid_data_g,valid_data_b)).reshape(valid_data_r.shape[0],-1)



train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)
valid_data = pd.DataFrame(valid_data)
'''
valid_data.to_csv('./data/preprocessed/valid_data.csv',index=False)
test_data.to_csv('./data/preprocessed/test_data.csv',index=False)
train_data.to_csv('./data/preprocessed/train_data.csv',index=False)
'''
train_data = np.array(train_data).reshape(train_data.shape[0],32,32,3)
test_data = np.array(test_data).reshape(test_data.shape[0],32,32,3)
valid_data = np.array(valid_data).reshape(valid_data.shape[0],32,32,3)

train_data = rgb2gray(train_data)
test_data = rgb2gray(test_data)
valid_data = rgb2gray(valid_data)


train_data = train_data.reshape(train_data.shape[0],-1)
test_data = test_data.reshape(test_data.shape[0],-1)
valid_data = valid_data.reshape(valid_data.shape[0],-1)


train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)
valid_data = pd.DataFrame(valid_data)

valid_data.to_csv('./data/preprocessed/grey_data/valid_data.csv',index=False)
test_data.to_csv('./data/preprocessed/grey_data/test_data.csv',index=False)
train_data.to_csv('./data/preprocessed/grey_data/train_data.csv',index=False)




