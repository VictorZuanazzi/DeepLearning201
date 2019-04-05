"""
This module implements training and evaluation of a Convolutional Neural Network 
in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch.optim as optim
import torch

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

#set datatype to torch tensor
dtype = torch.FloatTensor

#use GPUs if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  """
  
  #calculates the mean accuracy over all predictions:
  accuracy = (predictions.argmax(axis=1) == targets.argmax(axis=1)).mean()

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on 
  the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  
  #all external parameters in a readable format.
  lr = FLAGS.learning_rate
  max_steps = FLAGS.max_steps
  batch_size = FLAGS.batch_size
  eval_freq = FLAGS.eval_freq
  data_dir = FLAGS.data_dir
  
  #fetch data
  data = cifar10_utils.get_cifar10(FLAGS.data_dir)
  n_classes = 10
  n_channels = 3
  
  #number of iterations to train the data in the whole dataset:
  n_iter = np.cel(len(data["train"].labels)/batch_size)
  
  #load model
  cnn_model = ConvNet(n_channels, n_classes)
  
  #Loss function
  loss_XE = torch.nn.CrossEntropyLoss()
  
  #keep track of how loss and accuracy evolves over time.
  l = np.zeros(max_steps) #loss on the batch
  acc = np.zeros(max_steps) #accuracy on the batch
  loss_all = np.zeros(ceil(max_steps/eval_freq)) #loss on the whole data
  acc_all = np.zeros(ceil(max_steps/eval_freq)) #accuracy on the whole data
  
  #Optimizer
  optmizer = optim.Adam(cnn_model.parameters(), lr=lr)
  
  #let's put some gpus to work!
  cnn_model.to(device)
  
  #index to keep track of the evaluations.
  eval_i = 0
  
  #Train shit
  for s in range(max_steps):
      
    #go through the entire dataset
    for n in range(n_iter):
          #fetch next batch of data
          X, y = data['train'].next_batch(batch_size)
          
          #use torch tensor + gpu 
          X = torch.from_numpy(X).type(dtype).to(device)
          y = torch.from_numpy(y).type(dtype).to(device)
          
          #reset gradient to zero before gradient descent.
          optmizer.zero_grad()
          
          #calculate loss
          probs = cnn_model.forward(X)
          loss = loss_XE(probs, y.argmax(dim=1)) 
          
          #backward propagation
          loss.backward()
          optmizer.step()
    
    #stores the loss and accuracy of the last batch for later analysis.
    l[s] = loss
    acc[s] = accuracy(probs, y)
  
    if s % eval_freq == 0:
        #calculate accuracy for the whole data set
        
        #fetch all the data
        X = data['train']._images
        y = data['train']._labels
      
        #use torch tensor + gpu 
        X = torch.from_numpy(X).type(dtype).to(device)
        y = torch.from_numpy(y).type(dtype).to(device)
        
        #actually calculates loss and accuracy
        probs = cnn_model.forward(X)
        loss_all[eval_i] = loss_XE(probs, y.argmax(dim=1)) 
        acc_all[eval_i] = accuracy(probs, y)
        
        #increments eval counter
        eval_i +=1
        
        
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

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
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
  FLAGS, unparsed = parser.parse_known_args()

  main()