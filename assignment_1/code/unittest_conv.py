# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:22:26 2019

@author: Victor Zuanazzi
"""
#unit test library
import unittest
import torch
#helpful libraries
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
#built functions for testing
from train_convnet_pytorch import accuracy

N_CLASSES = 5
BATCH_SIZE = 10

def roll(x, shift):  
    """cicular shift all number shift positions to the right
    pytorch has no implementation of np.roll"""
    return torch.cat((x[:,-shift:], x[:,:-shift]), dim = 1)

class testTrainConvnet(unittest.TestCase):

    def test_accuracyValue(self):
        """Test bug free value for random predictions and accuracies."""
        
        #onehot enconde the predictions (necessary for sklearn..accuracy_score)
        predictions = (torch.eye(N_CLASSES)[np.random.choice(N_CLASSES, BATCH_SIZE)])

        #create random labels
        targets = (torch.eye(N_CLASSES)[np.random.choice(N_CLASSES, BATCH_SIZE)])
        
        #trusted accuracy function
        acc = accuracy_score(targets.numpy(), predictions.numpy(), normalize = True)
        
        #truncates pytorch response to 5 digits 
        n_digits = 5
        acc_r = torch.round((torch.tensor(accuracy(predictions, targets))) * 10**n_digits) / (10**n_digits)
        
        self.assertEqual(acc, acc_r)
    
    def test_allMissClassified(self):
        """Test edge case where accuracy = 0."""
        
        #create random labels
        targets = (torch.eye(N_CLASSES)[np.random.choice(N_CLASSES, BATCH_SIZE)])
        
        #predicitions are targets shifited to the right so no classifications
        #match
        predictions = roll(targets, shift=1)
        
        self.assertEqual(0.0, accuracy(predictions, targets))
#        
    def test_totalAccuracy(self):
        """Test edge case where accuracy = 100%."""
        
        #create random labels
        targets = (torch.eye(N_CLASSES)[np.random.choice(N_CLASSES, BATCH_SIZE)])
        
        #predictions and targets are the same
        predictions = targets
        
        self.assertEqual(1.0, accuracy(predictions, targets))    

if __name__ == '__main__':
    unittest.main()
