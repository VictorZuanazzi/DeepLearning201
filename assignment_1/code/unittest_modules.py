# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:39:00 2019

@author: Victor Zuanazzi
"""
#unit test library
import unittest
import torch
#helpful libraries
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

from modules import LinearModule, ReLUModule, SoftMaxModule, CrossEntropyModule
import torch.nn as nn

N_CLASSES = 5
BATCH_SIZE = 10

def roll(x, shift):  
    """cicular shift all number shift positions to the right
    pytorch has no implementation of np.roll"""
    return torch.cat((x[:,-shift:], x[:,:-shift]), dim = 1)

class testLinModule(unittest.TestCase):

    def test_init_w(self):
        """asset the shape of the weight matrix of the linear module."""
        
        #input and output dimensions
        in_features = 10
        out_features = 3
        
        #numpy hand made linear module
        a = LinearModule(in_features, out_features)
        
        #gold-standard linear module
        b = nn.Linear(in_features, out_features)
        
        #shape of the weight matrix
        self.assertEqual(b.weight.shape, a.params["weight"].shape)
        self.assertEqual(b.weight.shape, a.grads["weight"].T.shape)
    
    
    def test_init_b(self):
        """asset the shapes of the bias of the linear module."""
        
        #input and output dimensions
        in_features = 10
        out_features = 3
        
        #numpy hand made linear module
        a = LinearModule(in_features, out_features)
        
        #shape of the weight matrix
        self.assertEqual((out_features,1), a.params["bias"].shape)
        self.assertEqual((1, out_features), a.grads["bias"].shape)
    

if __name__ == '__main__':
    unittest.main()
