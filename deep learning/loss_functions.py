# -*- coding: utf-8 -*-
"""
Created on Fri Jul 03 22:32:17 2015

@author: Inpiron
"""

import numpy as np

def svm(x, y):
  
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax(x, y):

  probs = np.exp(x - np.max(x, axis=1, keepdims=True))


  probs /= np.sum(probs, axis=1, keepdims=True)


  
  N = x.shape[0]
  
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N


  dx = probs.copy()

  dx[np.arange(N), y] -= 1

  dx /= N


  return loss, dx
  
def rms_cls(x, y):
  N = x.shape[0]

  y_1 = np.zeros_like(x)
  y_1[np.arange(N),y] = 1

  err = x-y_1

  loss = np.sqrt(np.sum(np.square(err)))
  loss = loss/N
  dx = err
  dx /= N

  return loss, dx
  
def rms_reg(x, y):
  N = x.shape[0]

  err = x-y
  loss = np.sqrt(np.sum(np.square(x-y)))
  loss = loss/N
  dx = err
  dx /= N
  return loss, dx

