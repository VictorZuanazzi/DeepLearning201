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
                                             size = (in_features, out_features))
    
    #Initialize biases self.params['bias'] with 0. 
    self.params["bias"] = np.zeros(out_features)
    
    #initialize gradients with zeros.
    self.grads["weight"]  = np.zeros((in_features, out_features))
    self.grads["bias"] = np.zeros(out_features)

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    Hint: You can store intermediate variables inside the object. They can
    be used in backward pass computation.                                                           #
    """
    
    #forward pass of the module. 
    out = np.matmul(x, self.params["weight"]) + self.params["bias"]
    
    #stores x for later backard pass
    self.x = x
    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module 
    """

    
    #linear backprop
    dx = np.matmul(self.params["weight"], dout)
    
    #dL/db 
    self.grads["bias"] = dout # np.sum(dout)
    
    #dL/dW
    x_shape = (self.params["weight"].shape[0],1)
    dout_shape = (1, self.params["weight"].shape[1])
    self.grads["weight"] = np.matmul(self.x.shape(x_shape), dout.shape(dout_shape))
    
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
    
    TODO:
    Implement backward pass of the module.
    """
    
    #element wise multiplication
    dx =  dout * self.poss  

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
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick 
    - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    #max trick for computation stability
    max_x = np.max(x)
    
    #numerator part of the softmax
    numerator = np.exp(x - max_x)
    
    #denominator sum of the softmax
    denominator = numerator.sum()
    
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
    
    TODO:
    Implement backward pass of the module.
    """
    
    #how does that work again?
    diag_out = np.eye(self.out.shape(0))*self.out
    
    dxdx_t = diag_out- np.einsum('ij, ik, -> ijk', self.out, self.out)
    
    #einsum magic
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
    
    TODO:
    Implement forward pass of the module. 
    """
    
    #cross entropy from the text book:
    out = y * -1*np.log(x)

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
