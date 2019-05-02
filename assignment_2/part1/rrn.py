# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:51:40 2019

@author: Victor Zuanazzi
"""

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

################################################################################

class RRN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(RRN, self).__init__()
        
        
        #try Stijn's script once it is working
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device
        
        #standard deviation for initizalization of weights
        std = 1/num_hidden

        #parameters of the Forget Gate
        self.Wfx = nn.Parameter(std*torch.randn((input_dim, num_hidden)))
        self.Wfh = nn.Parameter(std*torch.randn((num_hidden, num_hidden)))
        self.bf = nn.Parameter(std*torch.randn(num_hidden))
        
        #parameters of the Input Gate
        self.Wix = nn.Parameter(std*torch.randn((input_dim, num_hidden)))
        self.Wih = nn.Parameter(std*torch.randn((num_hidden, num_hidden)))
        self.bi = nn.Parameter(std*torch.randn(num_hidden))
        
        #paramters of the Modulation Gate
        self.Wgx = nn.Parameter(std*torch.randn((input_dim, num_hidden)))
        self.Wgh = nn.Parameter(std*torch.randn((num_hidden, num_hidden)))
        self.bg = nn.Parameter(std*torch.randn(num_hidden))
        
        #paramters of the Output Gate
        self.Wox = nn.Parameter(std*torch.randn((input_dim, num_hidden)))
        self.Woh = nn.Parameter(std*torch.randn((num_hidden, num_hidden)))
        self.bo = nn.Parameter(std*torch.randn(num_hidden))
        
        #hidden and cell states:
        self.c = torch.empty(batch_size, num_hidden, device = device)
        self.h = torch.empty(batch_size, num_hidden, device = device)
        
        #Linear layer
        self.Wph = nn.Parameter(std*torch.randn((num_hidden, num_classes)))
        self.bp = nn.Parameter(std*torch.randn(num_classes))

    def reset(self):
        
        #reset cell state
        self.c.detach_()
        nn.init.constant_(self.c, 0.0)
        
        #reset hidden state
        self.h.detach_()
        nn.init.constant_(self.h, 0.0)

    def forward(self, x):
        
        #reset cell and hidden state before starting the sequence
        self.reset()
        
        #forward pass through the sequence
        for t in range(self.seq_length):
            xt = x[:, t].view(self.batch_size, self.input_dim)
            
            #g = torch.relu(xt @ self.Wgx + self.h @ self.Wgh + self.bg) #eq 4
            i = torch.relu(xt @ self.Wix + self.h @ self.Wih + self.bi) #eq 5
            f = torch.relu(xt @ self.Wfx + self.h @ self.Wfh + self.bf) #eq 6
            #o = torch.relu(xt @ self.Wox + self.h @ self.Woh + self.bo) #eq 7
            
            self.c = self.c + i - f #g * i + self.c * f #eq 8
            self.h = torch.relu(self.c) #* o #eq 9
            
        #linear foward step
        p = self.h @ self.Wph + self.bp #eq 10
        
        return p
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        