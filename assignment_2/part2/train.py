# MIT License
#
# Copyright (c) 2017 Tom Runia
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
# http://karpathy.github.io/2015/05/21/rnn-effectiveness/
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

################################################################################


def char2hot(batch, vocab_size):
    """encodes characters as 1 hot encoded vectors"""
    
    #size of the batch after been one hot encodded.
    encodded_size = list(batch.shape)
    encodded_size.append(vocab_size)
    
    #create a tensor of zeros
    one_hot = torch.zeros(encodded_size, 
                        device = batch.device)
    
    #add the hotties!
    one_hot.scatter_(2, batch.unsqueeze(-1), 1)
    
    return one_hot

def accuracy(predictions, label, config):
    """computes accuracy as the average of the accuracies for all steps."""
    return torch.sum(predictions.argmax(dim=2) == label).to(torch.float32) / (config.batch_size * config.seq_length)

def sample_from_model(y, temperature):
    """sample a carachter from the model"""
    
    #get the distribution across carachters
    distribution = torch.softmax(y.squeeze()/temperature, dim=0)
    
    #sample one character from the distribution
    return torch.multinomial(distribution,1).item()
    

  
def text_generator_9000(model, seed, length, dataset, device, temperature):
    """Generates a text given a seed."""
    
    with torch.no_grad():
        
        #the seed is the very first leter of the text
        text = seed.view(-1).tolist()
        
        #convert seed into a one-hot representation
        seed = char2hot(seed, dataset.vocab_size)
        
        #forward pass through the model
        y, (h, c) = model(seed)
        
        #sample a character from the output
        next_char = sample_from_model(y[:,-1,:], temperature)
        
        text.append(next_char)
        
        #can this loop be prettier?
        for l in range(length -1):
            
            #one-hot encode the previous carachter
            x = char2hot(torch.tensor(next_char, 
                                      dtype = torch.long,
                                      device = device).view(1,-1), 
                        dataset.vocab_size)
            
            #forward pass through the model
            y, (h, c) = model(x, (h, c))
            
            #sample a character from the output
            next_char = sample_from_model(y, temperature)
            
            #append the char to the text
            text.append(next_char)
            
        #convert indexes into chars
        text = dataset.convert_to_string(text)
        return text
     

def train(config):

    # Initialize the device which to run the model on
    wanted_device = config.device.lower()
    if wanted_device == 'cuda':
        #check if cuda is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        #cpu is the standard option
        device = torch.device('cpu')
    
    #Prints the device in use, it may be different than the one requested.
    print(f"Device available: {device}")
        
    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  
    
    data_loader = DataLoader(dataset, config.batch_size, num_workers = 0)


    # Initialize the model that we are going to use
    model = TextGenerationModel(batch_size = config.batch_size, 
                                seq_length = config.seq_length, 
                                vocabulary_size = dataset.vocab_size,
                                lstm_num_hidden = config.lstm_num_hidden, 
                                lstm_num_layers = config.lstm_num_layers, 
                                device = device)
    
    model = model.to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), 
                                    lr = config.learning_rate)
    
    #keeps the progress for later analysis
    train_acc = np.zeros(int(config.train_steps)+1)
    train_text = []
    
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        #get a batch with sentences.
        x = torch.stack(batch_inputs, dim=1).to(device)
        
        #create one_hot representations of the carachteer
        x = char2hot(x, dataset.vocab_size)
        
        #get labels
        y_true = torch.stack(batch_targets, dim=1).to(device)
        
        #forward pass
        y_pred, _ = model(x)
        loss = criterion(y_pred.transpose(2,1), y_true)
        
        #backward prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_acc[step] = accuracy(y_pred, y_true, config)
        
        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:
            
#            print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M")}] Train Step {step}/{config.train_steps}, Batch Size = {config.batch_size}, Examples/Sec = {examples_per_second}, Accuracy = {train_acc[step]}, Loss = {loss}")
             
            print(f"Train Step {step}/{config.train_steps}, Examples/Sec = {examples_per_second}, Accuracy = {train_acc[step]}, Loss = {loss}")
#            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
#                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
#                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
#                    config.train_steps, config.batch_size, examples_per_second,
#                    train_acc[step], loss))
            
            #save model, just in case
            torch.save(model, config.txt_file + "_model.pt")

        if step % config.sample_every == 0:
            # Generate some sentences by sampling from the model
            random_seed = torch.randint(low = 0,
                                        high = dataset.vocab_size, 
                                        size = (1, 1), 
                                        dtype=torch.long, 
                                        device=device)
            
            train_text.append("Training "+ str(step))
            train_text.append(text_generator_9000(model = model, 
                                                  seed = random_seed, 
                                                  length = config.seq_length, 
                                                  dataset = dataset, 
                                                  device = device, 
                                                  temperature = config.temperature))
            
            print(f"{step} Text Sample: \n{train_text[-1]}")
            

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            print(f"End of training: {step}")
            break
        
    print('Done training.')
    
    #Save the final model
    torch.save(model, config.txt_file + "_model.pt")
    np.save("train_acc", train_acc)
    
    temps = [0.01, 0.5, 1.0, 2.0, 10.0]
    
    for t in temps:
        for i in range(dataset.vocab_size):
            seed = torch.randint(low = i,
                                        high = i+1, 
                                        size = (1, 1), 
                                        dtype=torch.long, 
                                        device=device)
            
            train_text.append("Generation, temperature: "+ str(t) + "leter: " + str(i) + " = " + dataset.convert_to_string([i]))
            train_text.append(text_generator_9000(model = model, 
                                                  seed = seed, 
                                                  length = config.seq_length, 
                                                  dataset = dataset, 
                                                  device = device, 
                                                  temperature = config.temperature))
    
    with open(config.output_file, 'w', encoding="utf-8", errors="surrogateescape") as file:
        file.writelines(["%s\n" % item  for item in train_text])


 ################################################################################
 ################################################################################

def print_config(config):
  """
  Prints all entries of the config.
  """
  for key, value in vars(config).items():
    print(key + ' : ' + str(value))

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=False,
                        default = './Shakespeare/Shakespeare.txt',
                        help="Path to a .txt file to train on")
    parser.add_argument('--output_file', type=str, required=False,
                        default = './Shakespeare/Shakespeare_generated.txt',
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, 
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, 
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, 
                        help='Number of LSTM layers in the model')
    

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, 
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='Device to used "cuda" or "cpu"')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, 
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, 
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, 
                        help='Dropout keep probability')
    parser.add_argument('--train_steps', type=int, default= 300,#1e6, 
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, 
                        help='--')
    parser.add_argument('--temperature', type = float, default=1.0,
                        help='sampling temperature')

    # Misc params
    parser.add_argument('--summary_path', type=str, 
                        default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, 
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, 
                        help='How often to sample from the model')

    config = parser.parse_args()
    
    print_config(config)
        

    # Train the model
    train(config)
