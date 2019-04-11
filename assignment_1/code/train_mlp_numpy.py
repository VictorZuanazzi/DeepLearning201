"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

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
  
  TODO:
  Implement accuracy computation.
  """
  accuracy = (predictions.argmax(axis=1) == targets.argmax(axis=1)).mean()

  return accuracy

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

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []
  
  #to make things easier to read from here onwards.
  lr = FLAGS.learning_rate
  max_steps = FLAGS.max_steps
  batch_size = FLAGS.batch_size
  eval_freq =  FLAGS.eval_freq
  data_dir = FLAGS.data_dir 
  
  #load cifar10
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
  
  #get the number of classes
  n_classes = y_test.shape[1]
  
  #initialize the model
  MLP_model = MLP(n_inputs = flat_image, 
                  n_hidden = dnn_hidden_units,
                  n_classes = n_classes)
  
  #loads the loss function
  loss_func = CrossEntropyModule()
  
  #metrics to keep track during training.
  acc_train = []
  acc_test = []
  loss_train = []
  loss_test = []
  
  
  for step in range(max_steps):
      
      #loads minibatch
      X, y = cifar10['train'].next_batch(batch_size)
      
      #reshape the images
      X = X.reshape((batch_size, flat_image))
      
      #forward pass
      out = MLP_model.forward(X)
      
      #calculates the loss
      loss_t = loss_func.forward(out, y)
      
      #get gradient of the loss
      l_grad = loss_func.backward(out, y)
      
      #back prop on the model
      MLP_model.backward(l_grad)
      
      #perform SGD
      for layer in MLP_model.layer:
          layer.params["weight"] -= lr*layer.grads["weight"]
          layer.params["bias"] -= lr*layer.grads["bias"]
    

      if (step%eval_freq == 0) | (step == max_steps -1):
          print(f"step:{step}")
          
          #keep metrics on training set:
          loss_train.append(loss_t)
          acc_train.append(accuracy(out, y))
          print(f"train performance: acc = {acc_train[-1]}, loss = {loss_train[-1]}")
          
          #keep metrics on the test set:
          out = MLP_model.forward(x_test)
          loss_test.append(loss_func.forward(out, y_test))
          acc_test.append(accuracy(out, y_test))
          print(f"test performance: acc = {acc_test[-1]}, loss = {loss_test[-1]}")
          
          
  #finished training:
  print("saving results in folder...")
  np.save("np_loss_train", loss_train)
  np.save("np_accuracy_train", acc_train)
  np.save("np_loss_test", loss_test)
  np.save("np_accuracy_test", acc_test)
  
  print("saving model in folder")
  np.save("MLP_np_model", MLP_model)
  

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
  print("Step 1 of project Take Over the World: distinguish between cats and dogs")

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)
    
  # Run the training operation
  train()
  print("Training finished")
  print("Note to self, this dum ass coding you did an aweful job.")
  print("Find better implementation somewhere...")

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
  FLAGS, unparsed = parser.parse_known_args()

  main()