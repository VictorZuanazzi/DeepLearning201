# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 22:32:07 2019

@author: Victor Zuanazzi
"""
class net_params():
    def __init__(self, init = True):
        self.accuracy = 0
        
        if init:
            #initlialize individual with learnable mutation parameters
            self.dnn_hidden_units_choice = ['100', '200', '300' '100,10', '50,10', '10,10']
            self.hidden_units = np.random.choice(self.dnn_hidden_units_choice)
            self.hu_p = np.random.uniform(0.1, 1)
            self.lr = np.random.uniform(2e-5, 2e-3)
            self.lr_sigma = np.random.uniform(1e-6,1e-3)
            self.max_steps = np.random.randint(1000, 5000)
            self.ms_sigma = np.random.randint(100, 1000)
            self.batch_size = np.random.randint(100, 1000)
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

def optimize_MLP():
    
    #number of trials
    epochs = 3
    num_nets = 3
    nets = [net_params() for net in range(num_nets)]
    
    #shwallow archtecture
    nets[0].dnn_hidden_units_choice = ['100', '200', '300']
    
    #deep archtecture
    nets[-1].dnn_hidden_units_choice = ['10,10,10','50,20,10', '10,10,10,10,10']
    
    #store the accuracy
    acc = []
    
    for epoch in range(epochs):
        accs = []
        for n, net in enumerate(nets):
            print(f"ind {n} epoc {epoch}")
            
            #pass the parameters to the flags
            FLAGS.dnn_hidden_units = net.hidden_units
            FLAGS.learning_rate = net.lr
            FLAGS.max_steps = net.max_steps
            FLAGS.batch_size = net.batch_size
        
            print_flags()
            net.accuracy = train()[-1]
            
            accs.append(net.accuracy)
        
        best_net = np.argmax(accs)
        print(f"accuracies: accs")
        
        for n, net in enumerate(nets):
            if n == best_net:
                #the best one remains untouched
                continue
            #sex time!
            if np.random.uniform() < .5:
                net.sex(nets[best_net])
            else:
                net.sex(np.random.choice(nets))
            
            #mutatation
            net.mutate()
        
    
    FLAGS.dnn_hidden_units = nets[best_net].hidden_units
    FLAGS.learning_rate = nets[best_net].lr
    FLAGS.max_steps = nets[best_net].max_steps
    FLAGS.batch_size = net.batch_size
    nets[best_net].batch_size
    
    print("best net")
    print("accuracies: ", nets[best_net].accuracy)
    print(f"best setting: ")
    print(f"    dnn_hidden_units_s:  {nets[best_net].hidden_units}")
    print(f"    learning reate: {nets[best_net].lr}")
    print(f"    max_steps:  {nets[best_net].max_steps}")
    print(f"    batch size: {nets[best_net].batch_size}")    