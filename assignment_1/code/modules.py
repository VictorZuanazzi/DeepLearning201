"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample
    """

    self.params = {'weight': None, 'bias': None}
    self.grads = {'weight': None, 'bias': None}
    
    #Initialize weights self.params['weight'] using normal distribution with 
    #mean = 0 and std = 0.0001.
    self.params["weight"] = np.random.normal(loc = 0.0, 
                                             scale = 0.001, 
                                             size = (out_features, in_features))
    
    #Initialize biases self.params['bias'] with 0. 
    self.params["bias"] = np.zeros((out_features,1))
    
    #initialize gradients with zeros.
    self.grads["weight"]  = np.zeros((out_features, in_features))#np.zeros((in_features, out_features))
    self.grads["bias"] = np.zeros((out_features, 1))#np.zeros((1, out_features))

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module, shape(batch_size, features)
    Returns:
      out: output of the module
    
    Hint: You can store intermediate variables inside the object. They can
    be used in backward pass computation.                                                           #
    """
    
    #forward pass of the module. 
    out = np.matmul(self.params["weight"], x.T) + self.params["bias"]
    
    #stores x for later backard pass
    self.x = x
    return out.T

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module 
    """
    #linear backprop
    dx = np.matmul(dout, self.params["weight"])
    
    #dL/db 
    grad_b_shape = self.grads["bias"].shape
    self.grads["bias"] = np.reshape(dout.sum(axis=0), grad_b_shape)

    #dL/dW
    self.grads["weight"] = np.matmul(dout.T, self.x)
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    Hint: You can store intermediate variables inside the object. They can be 
    used in backward pass computation.                                                           #
    """
    
    #ReLU module
    out = np.maximum(x, 0)
    
    #save wich arguments are >0 and which aren't for a cheaper backwards pass.
    self.pos = x > 0

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    """
    
    #element wise multiplication
    dx =  dout * self.pos 

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    #max trick for computation stability
    max_x = x.max(axis=1)
    max_x = max_x.reshape((max_x.shape[0], 1))
    
    #numerator part of the softmax
    numerator = np.exp(x - max_x)
    
    #denominator sum of the softmax
    denominator = numerator.sum(axis=1)
    denominator = denominator.reshape((denominator.shape[0], 1))
    
    #softmax
    #saves output for a cheaper backwards pass
    self.out = numerator/denominator

    return self.out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    """
    
    #crazy shit over here!
    
    #explicit dimensions
    batch_size = self.out.shape[0] #mini batch size
    dim = self.out.shape[1] #feature dimension

    #creates a tensor with self.out elements in the diagonal
    diag_xN = np.zeros((batch_size, dim, dim))
    ii = np.arange(dim)
    diag_xN[:, ii, ii] = self.out
    
    #einstein sum convention to the rescue! :sunglasses:
    #first we calculate the dx/d\tilde{x} 
    dxdx_t = diag_xN - np.einsum('ij, ik -> ijk', self.out, self.out)
    
    
    dx = np.einsum('ij, ijk -> ik', dout, dxdx_t)

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    """
    
    #cross entropy from the text book:
    out = np.sum(y * (-1)*np.log(x), axis=1).mean()

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    """

    dx = -np.divide(y,x)/len(y)

    return dx
