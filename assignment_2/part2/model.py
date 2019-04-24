# MIT License
#
# Copyright (c) 2017 Tom Runia
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


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size=64, seq_length=30, vocabulary_size=10,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cpu'):

        super(TextGenerationModel, self).__init__()
        
        #load inputs:
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.device = device
        self.batch_first = True
        
        #initialize the LSTM cell
        self.lstm = nn.LSTM(input_size = self.vocabulary_size,
                            hidden_size = self.lstm_num_hidden,
                            num_layers = self.lstm_num_layers,
                            batch_first = self.batch_first)
        
        #initizalize the linear leayer
        self.linear = nn.Linear(in_features = self.lstm_num_hidden,
                                out_features = self.vocabulary_size,
                                bias = True)

    def forward(self, x, last_states = None):
        """x: input, torch.tensor 
        last_states: (last_hidden, last_cell) tuple(torch.tensor, torch.tensor)"""
        
        #LSTM forward
        all_hidden, (last_hidden, last_cell) = self.lstm(x, last_states)
        
        #Linear layer forward
        out = self.linear(all_hidden)
        
        return out, (last_hidden, last_cell)
        
        
        
