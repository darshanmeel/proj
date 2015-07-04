# -*- coding: utf-8 -*-
"""
Created on Fri Jul 03 22:35:02 2015

@author: Inpiron
"""
import numpy as np
import math
from active_functions import *
from loss_functions import *

class autoencoder_single_hidden_layer:
    def __init__(self,n_hidden,n_input,n_out,fnc = 'sigmoid',loss_fnc= softmax,batchsize = 10,epochs = 1,learning_rate = 0.1,reg = 0.0,momentum = 0.0):
                
        self.nn = {}
        self.nn['batchsize'] = batchsize
        self.nn['epochs'] = epochs
        self.nn['learning_rate'] = learning_rate
        self.nn['reg'] = reg
        self.nn['momentum'] = momentum
        self.nn['loss_fnc'] = loss_fnc
        self.nn['w1'] = np.random.randn(n_hidden*n_input).reshape(n_input,n_hidden)/math.sqrt(n_hidden*n_input)
        self.nn['b1'] = np.zeros(n_hidden).reshape(n_hidden)
        
        self.nn['w2'] = np.random.random(n_hidden*n_out).reshape(n_hidden,n_out)/math.sqrt(n_hidden*n_out)
        self.nn['b2'] = np.zeros(n_out).reshape(n_out)
        
        self.nn['dw1'] = np.zeros_like(self.nn['w1'])
        self.nn['db1'] = np.zeros_like(self.nn['b1'])
        
        self.nn['dw2'] = np.zeros_like(self.nn['w2'])
        self.nn['db2'] = np.zeros_like(self.nn['b2'])
        
        self.nn['p_dw1'] = self.nn['dw1']
        self.nn['p_db1'] = self.nn['db1']
        
        self.nn['p_dw2'] = self.nn['dw2']
        self.nn['p_db2'] = self.nn['db2']
        


    def fwd(self,x):
        # first layer
        w1 = self.nn['w1']
        b1 = self.nn['b1']

        dout1,dcache1 = fully_connected_fwd(x,w1,b1)

        

        
        dout2,dcache2 = sigm_forward(dout1)

        #Second layer
        
        w2 = self.nn['w2']
        b2 = self.nn['b2']

        dout3,dcache3 = fully_connected_fwd(dout2,w2,b2)

        self.nn['dcache1'] = dcache1
        self.nn['dcache2'] = dcache2
        self.nn['dcache3'] = dcache3
        self.nn['dout2'] = dout2

        
        return (dout3)
        
    def backprop(self,loss):
        
        #second layer

        dcache3 = self.nn['dcache3']


        dx2,dw2,db2 = fully_connected_bkwd(loss,dcache3)

         
        self.nn['dw2'] = dw2
        self.nn['db2'] = db2
        
        
        #first layer
        dcache1 = self.nn['dcache1']
        dcache2 = self.nn['dcache2']
        
        dx3 = sigm_backward(dx2,dcache2)

        dx4,dw4,db4 = fully_connected_bkwd(dx3,dcache1)

        
        self.nn['dw1'] = dw4
        self.nn['db1'] = db4
        
    def update_weights(self):
        #use regularization
        
        self.nn['dw1'] = self.nn['dw1'] + self.nn['w1'] * self.nn['reg']
        self.nn['dw2'] = self.nn['dw2'] + self.nn['w2'] * self.nn['reg']
        
        #use momentum
        self.nn['dw1'] = self.nn['momentum']*self.nn['p_dw1'] - self.nn['learning_rate']*self.nn['dw1']
        self.nn['db1'] = self.nn['momentum']*self.nn['p_db1'] - self.nn['learning_rate']*self.nn['db1']
        
        self.nn['dw2'] = self.nn['momentum']*self.nn['p_dw2'] - self.nn['learning_rate']*self.nn['dw2']
        self.nn['db2'] = self.nn['momentum']*self.nn['p_db2'] - self.nn['learning_rate']*self.nn['db2']
        

        self.nn['p_dw1'] = self.nn['dw1']
        self.nn['p_db1'] = self.nn['db1']
        
        self.nn['p_dw2'] = self.nn['dw2']
        self.nn['p_db2'] = self.nn['db2']
        
        self.nn['w1'] = self.nn['w1'] +  self.nn['dw1'] * self.nn['learning_rate'] 
        self.nn['b1'] = self.nn['b1'] +  self.nn['db1']* self.nn['learning_rate']
        
        self.nn['w2'] = self.nn['w2'] +  self.nn['dw2']* self.nn['learning_rate'] 
        self.nn['b2'] = self.nn['b2'] +  self.nn['db2']* self.nn['learning_rate']
        
    def predict(self,dt):
        dout4  = self.fwd(dt) 
        return (dout4,self.nn['dout2'])
        
    def fit(self,train_data,test_data = None):
        N = train_data.shape[0]
        batchsize = self.nn['batchsize']
        epochs = self.nn['epochs']
        print epochs,N,batchsize
        #shuffle train data
        
        random_indexes = np.arange(N)
        #np.random.shuffle(random_indexes)

        train_data = train_data[random_indexes]
        print train_data.shape
        loss_fnc = self.nn['loss_fnc']

        valid_err = []
        train_err = []
        for epoch in range(epochs):
            print 'start epoch ', epoch            
            for n in range(0,N,batchsize):
                ed = n+batchsize
                if ed > N:
                    ed = N

                td = train_data[n:ed,:]
                
                tc = td     

                dout4 = self.fwd(td)                
                loss,dx = loss_fnc(dout4,tc)

                self.backprop(dx)
                # update weights
                self.update_weights()
            if test_data is None:
                pass
            else:
                # train error
            
                dout4,_  = self.predict(train_data)         
                train_pred = dout4
                err = np.sqrt(np.sum(np.square(train_pred - train_data)))

                train_err.append(err)
                
                dout4,_ = self.predict(test_data)         
                test_pred = dout4
                err = np.sqrt(np.sum(np.square(test_pred - test_data)))
                valid_err.append(err)              
            print 'end epoch ', epoch
        return(train_err,valid_err)
if __name__ == '__main__':
    
    train_data = np.array([[0,0],[0,1],[1,0],[1,1]],dtype='float64')
    train_class = train_data
    test_data = train_data
    test_class = train_class
    n_input = train_data.shape[1]
    n_hidden = 2
    n_out = n_input
    nn1 = NN_single_hidden_layer(n_hidden,n_input,n_out,epochs = 1,batchsize=1,learning_rate=0.95,loss_fnc=rms_reg)
    nn1.fit(train_data,test_data)
    
    print
    print nn1.nn['w1']
    print
    print nn1.nn['b1']
    print
    print nn1.nn['w2']
    print
    print nn1.nn['b2']
    
    '''
    train_data = np.array([[0,0],[0,1]],dtype='float64')
    train_data = np.array([[0,0]],dtype='float64')
    train_class = np.array([0,1]).reshape(2,1)
    train_class = np.array([0]).reshape(1,1)
    test_data = train_data
    test_class = train_class
    n_input = train_data.shape[1]
    n_hidden = 2
    n_out = 1
    nn1 = NN_single_hidden_layer(n_hidden,n_input,n_out,epochs = 1,batchsize=1,learning_rate=1.0,loss_fnc=rms_reg)
    w1 = nn1.nn['w1'] 
    b1 = nn1.nn['b1']
    print w1
    print b1
    w1[0][0]=0.2
    w1[0][1]=0.3
    w1[1][0]=0.25
    w1[1][1]=-0.1
    b1[0] = -0.1
    b1[1] = -0.15
    print w1
    print b1
    
    w2 = nn1.nn['w2'] 
    b2 = nn1.nn['b2']
    print w2
    print b2
    w2[0]=-0.1
    w2[1]=0.2
    b2[0] = 0.05

    print w2
    print b2
    

    nn1.fit(train_data,train_class,test_data,test_class)
    
    w1 = nn1.nn['w1'] 
    b1 = nn1.nn['b1']
    print w1
    print b1
    
    w2 = nn1.nn['w2'] 
    b2 = nn1.nn['b2']
    print w2
    print b2
    '''
    

    

        
    
        
        
    