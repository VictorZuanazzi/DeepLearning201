# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:22:26 2019

@author: Victor Zuanazzi
"""
#unit test library
import unittest

#helpful libraries
import numpy as np
from sklearn.preprocessing import normalize

#built functions for testing
from train_convnet_pytorch import accuracy

N_CLASSES = 5
BATCH_SIZE = 10

class testTrainConvnet(unittest.TestCase):

    def test_accuracyValue(self):
        """Test bug free value for random predictions and accuracies."""
        
        #create random predictions
        predictions = np.random.normal(size = (BATCH_SIZE,N_CLASSES))
        predictions = normalize(predictions, axis=1, norm='l1')
        
        #create random labels
        targets = np.eye(N_CLASSES)[np.random.choice(N_CLASSES, BATCH_SIZE)]
        
        #implemented logic should work (in one liner)
        acc = np.multiply(predictions, targets).sum()/len(targets)
        
        self.assertEqual(acc, accuracy(predictions, targets))
        
    def test_zeroAccuracy(self):
        """Test edge case where accuracy = 0."""
        
        #create random predictions
        predictions = np.zeros(shape=(10,N_CLASSES))
        
        #create random labels
        targets = np.eye(N_CLASSES)[np.random.choice(N_CLASSES, BATCH_SIZE)]
        
        self.assertEqual(0.0, accuracy(predictions, targets))
        
    def test_totalAccuracy(self):
        """Test edge case where accuracy = 100%."""
        
        #create random labels
        targets = np.eye(N_CLASSES)[np.random.choice(N_CLASSES, BATCH_SIZE)]
        
        #predictions and targets are the same
        predictions = targets
        
        self.assertEqual(1.0, accuracy(predictions, targets))
    

if __name__ == '__main__':
    unittest.main()
