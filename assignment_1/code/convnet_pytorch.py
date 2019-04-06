"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
    """
    
    #stores the input and output sizes
    self.n_channels = n_channels
    self.n_classes = n_classes
    
    #set number of filters for each conv layer
    n_filters = {
            "conv1": 64,  "maxpool1": 64,
            "conv2": 128, "maxpool2": 128,
            "conv3": 256, "maxpool3": 256,
            "conv4": 512, "maxpool4": 512,
            "conv5": 512, "maxpool5": 512,
            "avgpool": 512
            }
    
    #set stride for convs and pooling layers
    stride = {
            "conv": 1,
            "maxpool": 2,
            "avgpool": 1,
            }
    
    #set padding
    p = 1
    p_avg = 0 #only used for the last avgpool layer.
    
    #kernel size:
    k = 3 
    k_avg = 1 #only used for the last avgpool layer.
    
    super(ConvNet, self).__init__()
    
    #conv1
    self.conv1 = torch.nn.Conv2d(in_channels = n_channels, 
                                 out_channels = n_filters["conv1"], 
                                 kernel_size = k, 
                                 stride = stride["conv"], 
                                 padding = p)   
    #batchnorm1
    self.norm1 = torch.nn.BatchNorm2d(n_filters["conv1"])
    
    #maxpool1
    self.pool1 = torch.nn.MaxPool2d(kernel_size = k,
                                   stride = stride["maxpool"],
                                   padding = p)
    #conv2
    self.conv2 = torch.nn.Conv2d(in_channels = n_filters["conv1"], 
                                 out_channels = n_filters["conv2"], 
                                 kernel_size = k, 
                                 stride = stride["conv"], 
                                 padding = p)
    #batchnorm2
    self.norm2 = torch.nn.BatchNorm2d(n_filters["conv2"])
    
    #maxpool2
    self.pool2 = torch.nn.MaxPool2d(kernel_size = k,
                                   stride = stride["maxpool"],
                                   padding = p)
    #conv3
    self.conv3_a = torch.nn.Conv2d(in_channels = n_filters["conv2"], 
                                 out_channels = n_filters["conv3"], 
                                 kernel_size = k, 
                                 stride = stride["conv"], 
                                 padding = p)  
    #batchnorm3
    self.norm3 = torch.nn.BatchNorm2d(n_filters["conv3"])
    
    self.conv3_b = torch.nn.Conv2d(in_channels = n_filters["conv3"], 
                                 out_channels = n_filters["conv3"], 
                                 kernel_size = k, 
                                 stride = stride["conv"], 
                                 padding = p)
    #maxpool3
    self.pool3 = torch.nn.MaxPool2d(kernel_size = k,
                                   stride = stride["maxpool"],
                                   padding = p)
    #conv4
    self.conv4_a = torch.nn.Conv2d(in_channels = n_filters["conv3"], 
                                 out_channels = n_filters["conv4"], 
                                 kernel_size = k, 
                                 stride = stride["conv"], 
                                 padding = p)  
    #batchnorm4
    self.norm4 = torch.nn.BatchNorm2d(n_filters["conv4"])
    
    self.conv4_b = torch.nn.Conv2d(in_channels = n_filters["conv4"], 
                                 out_channels = n_filters["conv4"], 
                                 kernel_size = k, 
                                 stride = stride["conv"], 
                                 padding = p)
    #maxpool4
    self.pool4 = torch.nn.MaxPool2d(kernel_size = k,
                                   stride = stride["maxpool"],
                                   padding = p)
    #conv5
    self.conv5_a = torch.nn.Conv2d(in_channels = n_filters["conv4"], 
                                 out_channels = n_filters["conv5"], 
                                 kernel_size = k, 
                                 stride = stride["conv"], 
                                 padding = p)  
    #batchnorm5
    self.norm5 = torch.nn.BatchNorm2d(n_filters["conv5"])
    
    self.conv5_b = torch.nn.Conv2d(in_channels = n_filters["conv5"], 
                                 out_channels = n_filters["conv5"], 
                                 kernel_size = k, 
                                 stride = stride["conv"], 
                                 padding = p)
    #maxpool5
    self.pool5 = torch.nn.MaxPool2d(kernel_size = k,
                                   stride = stride["maxpool"],
                                   padding = p)
    
    #avgpool
    self.pool6 = torch.nn.AvgPool2d(kernel_size = k_avg,
                                   stride = stride["avgpool"],
                                   padding = p_avg)
    
    #linear layer (fully connected layer)
    self.fc = torch.nn.Linear(n_filters["conv5"], n_classes)


  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed 
    through several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """

    #All conv blocks consist of 2d-convolutional layer followed by Batch 
    #normalization layer and ReLu layer.
    
    #conv1 + batchnorm1
    x = torch.nn.functional.relu(self.norm1(self.conv1(x)))
    
    #maxpool1
    x = self.pool1(x)
    
    #conv2 + batchnorm2
    x = torch.nn.functional.relu(self.norm2(self.conv2(x)))
    
    #maxpool2
    x = self.pool2(x)
    
    #conv3 + batchnorm3
    x = torch.nn.functional.relu(self.norm3(self.conv3_a(x)))
    x = torch.nn.functional.relu(self.norm3(self.conv3_b(x)))
    
    #maxpool3
    x = self.pool3(x)
    
    #conv4 + batchnorm4
    x = torch.nn.functional.relu(self.norm4(self.conv4_a(x)))
    x = torch.nn.functional.relu(self.norm4(self.conv4_b(x)))
    
    #maxpool4
    x = self.pool4(x)
    
    #conv5 + batchnorm5
    x = torch.nn.functional.relu(self.norm5(self.conv5_a(x)))
    x = torch.nn.functional.relu(self.norm5(self.conv5_b(x)))
    
    #maxpool5
    x = self.pool5(x)
    
    #avgpool
    x = self.pool6(x)
    
    #fully connected layer
    out = torch.nn.functional.softmax(self.fc(x.view(x.shape[0], -1))) #self.fc(x.view(x.shape[0], -1)) #
    

    return out
