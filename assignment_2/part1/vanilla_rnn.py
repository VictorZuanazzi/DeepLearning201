################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        
        #Try script from Stijn after this thing is workin...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device
        
        #scale for initializing the weights
        #inspired in how torch initializes its vectors
        std = 1/num_hidden
        
        #Initialize Weight matrices
        self.Wxh = nn.Parameter(std*torch.randn((input_dim, num_hidden)))
        self.Whh = nn.Parameter(std*torch.randn((num_hidden, num_hidden)))
        self.Whp = nn.Parameter(std*torch.randn((num_hidden, num_classes)))
        
        #Initialize Biases
        self.bh = nn.Parameter(std*torch.randn(num_hidden))
        self.bp = nn.Parameter(std*torch.randn(num_classes))
        
        #hidden state
        self.h = torch.empty(batch_size, num_hidden, device=device)
        

    def forward(self, x):
        
        #reset hidden state t0 zero before the forward pass
        self.h.detach_()
        nn.init.constant_(self.h, 0.0)
        
        #forward through all steps of the sequence
        for t in range(self.seq_length):
            xt = x[:,t].view(-1, self.input_dim)
            hx = xt @ self.Wxh
            hh = self.h @ self.Whh
            self.h = torch.tanh(hx + hh + self.bh)
         
        #final linear layer
        out = self.h @ self.Whp + self.bp
        
        return out
