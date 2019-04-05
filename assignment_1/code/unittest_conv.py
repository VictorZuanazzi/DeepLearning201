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
from sklearn.metrics import accuracy_score
#built functions for testing
from train_convnet_pytorch import accuracy

N_CLASSES = 5
BATCH_SIZE = 10

class testTrainConvnet(unittest.TestCase):

    def test_accuracyValue(self):
        """Test bug free value for random predictions and accuracies."""
        
        #create random predictions approximating a softmax
        predictions = np.random.uniform(size = (BATCH_SIZE,N_CLASSES))**2
        predictions = normalize(predictions, axis=1, norm='l1')
        
        #onehot enconde the predictions (necessary for sklearn..accuracy_score)
        predictions = (predictions == predictions.max(axis=1)[:,None]).astype(int)

        #create random labels
        targets = (np.eye(N_CLASSES)[np.random.choice(N_CLASSES, BATCH_SIZE)]).astype(int)
        
        #trusted accuracy function
        acc = accuracy_score(targets, predictions, normalize = True)
        
        self.assertEqual(acc, accuracy(predictions, targets))
    
    def test_allMissClassified(self):
        """Test edge case where accuracy = 0."""
        
        #create random labels
        targets = np.eye(N_CLASSES)[np.random.choice(N_CLASSES, BATCH_SIZE)]
        
        #predicitions are targets shifited to the right so no classifications
        #match
        predictions = np.roll(targets, shift=1, axis=1)
        
        self.assertEqual(0.0, accuracy(predictions, targets))
#        
    def test_totalAccuracy(self):
        """Test edge case where accuracy = 100%."""
        
        #create random labels
        targets = np.eye(N_CLASSES)[np.random.choice(N_CLASSES, BATCH_SIZE)]
        
        #predictions and targets are the same
        predictions = targets
        
        self.assertEqual(1.0, accuracy(predictions, targets))    

if __name__ == '__main__':
    unittest.main()
