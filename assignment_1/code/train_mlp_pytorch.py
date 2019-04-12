"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch

from EA_opt import net_params

# Default constants
DNN_HIDDEN_UNITS_DEFAULT =  '200' #'100'
LEARNING_RATE_DEFAULT = 2e-4 #2e-3
MAX_STEPS_DEFAULT = 3000 #1500
BATCH_SIZE_DEFAULT = 1000 #200
EVAL_FREQ_DEFAULT = 100
AUTO_OPT_DEFAUT = True

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

#set datatype to torch tensor
dtype = torch.FloatTensor

#use GPUs if available
device = torch.device('cpu') #  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float torch array of size [batch_size, n_classes]
    labels: 2D int torch array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """
  #calculates the mean accuracy over all predictions:
  accuracy = (predictions.argmax(dim=1) == targets.argmax(dim=1)).type(dtype).mean()

  return accuracy.item() #returns the number instead of the tensor.


def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the 
  whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  torch.manual_seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []
    
  #for readability's sake
  lr = FLAGS.learning_rate
  max_steps =  FLAGS.max_steps
  batch_size = FLAGS.batch_size
  eval_freq = FLAGS.eval_freq
  data_dir = FLAGS.data_dir    

  cifar10 = cifar10_utils.get_cifar10(data_dir)
  
  #load test data
  x_test = cifar10["test"].images
  y_test = cifar10["test"].labels
  
  #get the dimensions of the data
  in_dim = x_test.shape
  n_samples = in_dim[0]
  height = in_dim[1]
  width = in_dim[2]
  channels = in_dim[3]
  flat_image = height * width * channels
  
  #reshape the test data so the MLP can read it.
  x_test = x_test.reshape((n_samples, flat_image))
  
  #tranform np arrays into torch tensors
  x_test = torch.tensor(x_test, requires_grad=False).type(dtype).to(device)
  y_test = torch.tensor(y_test, requires_grad=False).type(dtype).to(device)
  
  #get the number of classes
  n_classes = y_test.shape[1]
  
  #initialize the model
  MLP_model = MLP(n_inputs = flat_image, 
                  n_hidden = dnn_hidden_units,
                  n_classes = n_classes)
  
  #create loss function
  loss_func = torch.nn.CrossEntropyLoss()
  
  #loads Adam optimizer
  optimizer = torch.optim.Adam(MLP_model.parameters(), lr=lr)
  
  #metrics to keep track during training.
  acc_train = []
  acc_test = []
  loss_train = []
  loss_test = []
  
  for step in range(max_steps):
      
      #load minibatch
      X, y = cifar10['train'].next_batch(batch_size)
      
      #reshape images into a vector
      X = X.reshape((batch_size, flat_image))
      
      #use torch tensor + gpu 
      X = torch.from_numpy(X).type(dtype).to(device)
      y = torch.from_numpy(y).type(dtype).to(device)
      
      #set optimizer gradient to zero
      optimizer.zero_grad()
      
      #forward pass
      out = MLP_model.forward(X)
      
      #compute loss
      loss_t = loss_func(out, y.argmax(dim=1))
      
      #backward propagation
      loss_t.backward()
      optimizer.step()
      
      if (step%eval_freq == 0) | (step == max_steps -1):
          print(f"step:{step}")
          
          #keep metrics on training set:
          loss_train.append(loss_t)
          acc_train.append(accuracy(out, y))
          print(f"train performance: acc = {acc_train[-1]}, loss = {loss_train[-1]}")
          
          #keep metrics on the test set:
          out = MLP_model.forward(x_test)
          loss_test.append(loss_func.forward(out, y_test.argmax(dim=1)))
          acc_test.append(accuracy(out, y_test))
          print(f"test performance: acc = {acc_test[-1]}, loss = {loss_test[-1]}")
          
          #pytorch breaks here for some reason...
#          if len(loss_train)> 10:
#              #no changes in the past 10 evaluations
##              if (np.mean(loss_train[-10:-5]) - np.mean(loss_train[-5:])) < 1e-7:
##                  print("Early Stop")
##                  break  
  return acc_test
      
      
  #finished training:
  path = "./torch results/"
  print("saving results in folder...")
  np.save(path + "torch_loss_train", loss_train)
  np.save(path + "torch_accuracy_train", acc_train)
  np.save(path + "torch_loss_test", loss_test)
  np.save(path + "torch_accuracy_test", acc_test)
  
  print("saving model in folder")
  np.save(path+"MLP_torch_model", MLP_model)
  return acc_test

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()
  print("Step 1 of project Take Over the World (retrial) : distinguish between cats and dogs")

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)
    
  if FLAGS.auto_opt:
      optimize_MLP()
  
  #refit
  _ = train()

  # Run the training operation
  train()
  print("Training finished")
  print("Note to self, do not relly on human doing your code")
  print("Find better implementation somewhere...")     
  
def optimize_MLP():
    
    #number of trials
    epochs = 1#5
    #lisa trains about 10 nets per hour 
    num_nets = 1# 10
    nets = [net_params() for net in range(num_nets)]
    
    #shwallow archtecture
    nets[0].dnn_hidden_units_choice = ['1000', '500', '300']
    
    #deep archtecture
    nets[-1].dnn_hidden_units_choice = ['50,10,10','50,20,10', '40,30,20,10']
    
    #there isn't a best net yet
    best_net = -1
    
    for epoch in range(epochs):
        accs = []
        for n, net in enumerate(nets):
            
            #avoid retraining the best individual
            if n == best_net:
                continue         
            
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
        print(f"accuracies: {accs}")
        
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
        
    #reset flags to the best individual
    FLAGS.dnn_hidden_units = nets[best_net].hidden_units
    FLAGS.learning_rate = nets[best_net].lr
    FLAGS.max_steps = nets[best_net].max_steps
    FLAGS.batch_size = nets[best_net].batch_size
    
    print("best net")
    print("accuracies: ", nets[best_net].accuracy)
    print(f"best setting: ")
    print(f"    dnn_hidden_units_s:  {nets[best_net].hidden_units}")
    print(f"    learning reate: {nets[best_net].lr}")
    print(f"    max_steps:  {nets[best_net].max_steps}")
    print(f"    batch size: {nets[best_net].batch_size}")    

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--auto_opt', type = str, default = AUTO_OPT_DEFAUT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  

  main()