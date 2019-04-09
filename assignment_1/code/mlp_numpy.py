"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    """
       
    self.num_hlayers = len(n_hidden)
    self.num_layers = self.num_hlayers + 1
    
    #The linear layers are stored in a list
    self.layer = []
    
    #The activation layers are stored also in a list
    #The activation and the linear layers are in different lists to make 
    #them explicit in the foward pass.
    self.activation = []
    
    #the 0th layer receives is n_i x n_h(1)
    self.layer.append(LinearModule(size=(n_inputs, n_hidden[0])))
    self.activation.append(ReLUModule())
    
    #initializes the hidden layers
    for i in range(self.num_hlayers -1):
        self.layer.append(LinearModule(size=(n_hidden[i], n_hidden[i+1])))
        self.activation.append(ReLUModule())
        
    #the last layer must output the number of classes
    self.layer.append(LinearModule(size=(n_hidden[-1], n_classes)))
    self.activation.append(SoftMaxModule())
    
    if self.num_layers != len(self.layer):
        print("You've got a bug here! The sizes dont match.")
        print(f"num_layers = {self.num_layers} != len(layer) = {len(self.layer)}")
    
    
  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """
    
    #forward pass
    for i in range(self.num_layers):
        #linear layer
        x = self.layer[i].forward(x)
        
        #activation
        x = self.activation[i].forward(x)

    return x

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    #go through all layers backwards
    for i in range(self.num_layers -1, -1, -1):
        dout = self.activation[i].bakward(dout)
        dout = self.layer[i].bakward(dout)

    return
