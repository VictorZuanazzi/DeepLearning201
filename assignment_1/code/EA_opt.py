# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 22:32:07 2019

@author: Victor Zuanazzi
"""
import numpy as np

class net_params():
    def __init__(self, init = True):
        self.accuracy = 0
        
        if init:
            #initlialize individual with learnable mutation parameters
            self.dnn_hidden_units_choice = ['5000',
                                            '1000', 
                                            '1000,500',
                                            '1000,500,250', 
                                            '1000,500,250,100',
                                            '500,500',
                                            '500,500,250,250', 
                                            '500,500,250,250,100',
                                            '500,1000']
            #archtecture '1000,1000' memorizes the data
            
            self.hidden_units = np.random.choice(self.dnn_hidden_units_choice)
            self.hu_p = np.random.uniform(0.1, 1)
            self.lr = np.random.uniform(2e-6, 2e-4) #learning rate
            self.lr_sigma = np.random.uniform(1e-7,1e-5)
            self.max_steps = np.random.randint(1500, 5000)  #max steps
            self.ms_sigma = np.random.randint(100, 1000)
            self.batch_size = np.random.randint(100, 1000) #batch size
            self.bs_sigma = np.random.randint(10, 100)
        
    def set_params(self, hidden_units, lr, max_steps, batch_size): 
        self.accuracy = 0
        #externally set parameters
        self.hidden_units = hidden_units
        self.lr = lr
        self.max_steps = int(max_steps)
        self.batch_size =int(batch_size)
    
    def mutate(self):
        self.accuracy = 0
        #mutate individual
        self.s_sigma = .01
        self.epsilon = 1e-5
        
        #number of hidden units
        self.hu_p += np.random.normal(scale=self.s_sigma)
        self.hu_p = np.maximum(self.hu_p, self.epsilon)
        if np.random.uniform() < self.hu_p:
            self.hidden_units = np.random.choice(self.dnn_hidden_units_choice)
        
        #learning rate
        self.lr_sigma += np.random.normal(scale=self.s_sigma)
        self.lr_sigma = np.maximum(self.lr_sigma, self.epsilon)
        self.lr += np.random.normal(scale=self.lr_sigma)
        self.lr = np.maximum(self.lr, 1e-6)
        
        #max steps
        self.ms_sigma += np.random.normal(scale=self.s_sigma*100) 
        self.ms_sigma = np.maximum(self.ms_sigma, self.epsilon)
        self.max_steps += int(np.random.normal(scale=self.ms_sigma))
        
        #batch size
        self.bs_sigma += np.random.normal(scale=self.s_sigma*100)
        self.bs_sigma = np.maximum(self.bs_sigma, self.epsilon)
        self.batch_size += int(np.random.randint(100, 1000))
    
    def sex(self, partner):
        self.accuracy = 0
        #new individual is made together with a partner.
        
        #hidden units
        self.hu_p = np.random.uniform(self.hu_p, partner.hu_p)
        if np.random.uniform() < self.hu_p:
            self.hidden_units = partner.hidden_units
        
        #learning rate
        self.lr_sigma = np.random.uniform(self.lr_sigma, partner.lr_sigma)
        self.lr = np.random.uniform(self.lr, partner.lr)
        
        #max_steps
        self.ms_sigma = np.random.uniform(self.ms_sigma, partner.ms_sigma)
        self.max_steps = int(np.random.uniform(self.max_steps, partner.max_steps))
        
        #batch size
        self.bs_sigma = np.random.uniform(self.bs_sigma, partner.bs_sigma)
        self.batch_size = int(np.random.uniform(self.batch_size, partner.batch_size))

