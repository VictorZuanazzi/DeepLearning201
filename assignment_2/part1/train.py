################################################################################
# MIT License
# 
# Copyright (c) 2018
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

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

def accuracy(predictions, label, config):
    """computes accuracy as the average of the accuracies for all steps."""
    return torch.sum(predictions.argmax(dim=1) == label).to(torch.float) / (config.batch_size)


def train(config):
    
    config.model_type = config.model_type.lower()
    assert config.model_type in ('rnn', 'lstm')
    
    # Initialize the device which to run the model on
    wanted_device = config.device.lower()
    if wanted_device == 'cuda':
        #check if cuda is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        #cpu is the standard option
        device = torch.device('cpu')
        
    
    # Initialize the model that we are going to use    
    if config.model_type == 'rnn':
        model = VanillaRNN(seq_length = config.input_length,
                           input_dim = config.input_dim,
                           num_hidden = config.num_hidden,
                           num_classes = config.num_classes,
                           batch_size = config.batch_size,
                           device = device)
    elif config.model_type == 'lstm':
        model = LSTM(seq_length = config.input_length,
                       input_dim = config.input_dim,
                       num_hidden = config.num_hidden,
                       num_classes = config.num_classes,
                       batch_size = config.batch_size,
                       device = device)
        

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=0)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.RMSprop(model.parameters(), 
                                        lr=config.learning_rate)
        
    train_acc = np.zeros(config.train_steps+1)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        #batches to torch tensors
        x = torch.tensor(batch_inputs, dtype=torch.float, device=device)
        y_true = torch.tensor(batch_targets, dtype=torch.long, device=device)

        #Forward pass
        y_pred = model.forward(x)
        loss = criterion(y_pred, y_true)
        
        #Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        ############################################################################
        # QUESTION: what happens here and why?
        # clip_grad_norm() is a method to avoid exploding gradients. It clips 
        # gradients above max_norm to max_norm.
        #Deprecated, use clip_grad_norm_() instead
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        optimizer.step()
        
        train_acc[step] = accuracy(y_pred, y_true, config)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/(float(t2-t1) + 1e-6)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    train_acc[step], loss
            ))
            print(f"x: {x[0,:]}, y_pred: {y_pred[0,:].argmax()}, y_true: {y_true[0]}")

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')
    #Save the final model
    torch.save(model, config.model_type + "_model.pt")
    np.save("train_acc_" + config.model_type + str(config.input_length), train_acc)

 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda'")

    config = parser.parse_args()

    # Train the model
    train(config)