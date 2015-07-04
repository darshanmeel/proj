import numpy as np

# this will be used to do a some of all weights and data and bias. Actual activation functions wont be passed
# w x and b
def fully_connected_fwd(x, w, b):  
  out = x.reshape(x.shape[0], -1).dot(w) 
  out = out + b
  cache = (x, w, b)
  return out, cache
  
def fully_connected_bkwd(dout, cache):
  x, w, b = cache  
  dx = dout.dot(w.T).reshape(x.shape)  
  dw = x.reshape(x.shape[0], -1).T.dot(dout) 
  db = np.sum(dout, axis=0)


  return dx, dw, db

def sigm_forward(x):
    out = 1.0/(1  + np.exp(x*-1.0))
    cache = (x,out)
    return(out,cache)

def sigm_backward(res,cache):
    x,out = cache

    dx = res * out * (1.0 -out)

    return dx
    
def tanh_forward(x):
    epx = np.exp(x)
    enx = np.exp(-1.0*x)
    out = (epx - enx)/(epx + enx)
    cache = (x,out)
    return(out,cache)

def tanh_backward(res,cache):
    x,out = cache
    dx = res * 1- np.power(out,2)
    return dx
    
    
def relu_fwd(x):
  out = np.maximum(0, x)
  cache = x
  return out, cache


def relu_bkwd(res, cache):
  x = cache
  dx = np.where(x > 0, res, 0)
  return dx


def dropout_fwd(x, dropout_param):

  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    pass
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_bkwd(dout, cache):
  dropout_param, mask = cache
  mode = dropout_param['mode']
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx

