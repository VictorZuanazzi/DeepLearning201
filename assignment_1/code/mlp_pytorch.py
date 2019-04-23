"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
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
    
    super(MLP, self).__init__()
    
    #linear and activation layers are all in the same list
    self.layer = []
    
    #for elegance, we first all dimensions in n_hidden
    n_hidden.insert(0, n_inputs) 
    n_hidden.append(n_classes)
    
    #creates a list with all the layers
    for l in range(len(n_hidden)-1):
        self.layer.append(nn.BatchNorm1d(n_hidden[l]))
        self.layer.append(nn.Linear(n_hidden[l], n_hidden[l+1]))
        if l < len(n_hidden) - 2:
            #no ReLU is apllied in the last layer
            self.layer.append(nn.ReLU())
            self.layer.append(nn.Dropout())

    #creates model
    self.model = nn.Sequential(*self.layer)


  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """
    
    #just like magic!
    out = self.model(x)

    return out
