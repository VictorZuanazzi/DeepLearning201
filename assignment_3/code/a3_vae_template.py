import argparse
import os

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, in_dim=784):
        super().__init__()
        
        
        self.mean = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, z_dim))
        
        self.covariance = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, z_dim),
                                    nn.ReLU()) #relu is necessary std cannot be negative

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        mean = self.mean(input)
        
        std = self.covariance(input)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, out_dim = 784):
        super().__init__()
        
        self.generator = nn.Sequential(nn.Linear(z_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, out_dim),
                                       nn.Sigmoid())
        

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        
        mean = self.generator(input)

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, device = "cpu", im_dim=784):
        super().__init__()
        
        self.im_dim= im_dim
        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim).to(device)
        self.decoder = Decoder(hidden_dim, z_dim).to(device)
        self.device = device
        
    
        
    def elbo_loss(self, input, decoded, mean, std):
        
        
        epsilon = 1e-8 # computational stability
        
        decoded = decoded + epsilon
        
        # negative log bernoulli loss
        recontruction_loss = -1 * torch.sum(input * torch.log(decoded) + (1 - input) * torch.log(1 - decoded), 
                                        dim = 1)
        
        # KL divergence
        var = std.pow(2)
     
        KL = torch.sum(-torch.log(std + epsilon) + (var + mean.pow(2))/2 -1/2,
                       dim = 1)
        
        regularization_loss = KL        
        
        return torch.mean(recontruction_loss + regularization_loss, dim = 0)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        
        # flatten input
        input = input.view(-1, self.im_dim).to(self.device) 
        
        mean, std = self.encoder(input)
        
        # sample z using the reparametrization trick
        eps = torch.randn((1, self.z_dim), device = self.device)
        z = mean + std * eps
        
        # reconstruct inputs
        decoded = self.decoder(z)
        
        # calculate the ELBO loss
        average_negative_elbo = self.elbo_loss(input, decoded, mean, std)

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        
        # sample z from standard normal distribution
        z = torch.randn((n_samples, self.z_dim), device= self.device)
        
        # get sample images from decoder
        sampled_ims = self.decoder(z)
        
        # average
        im_means = sampled_ims.mean(dim=0)
        
        #dinamically determin the size of the image
        side = int(np.sqrt(self.im_dim))
        
        return sampled_ims.view(-1, 1, side, side), im_means.view(1, 1, side, side)


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = 0
    
    data_size = len(data)
    for i, batch in enumerate(data):
        #get the average elbo of the batch
        elbo = model(batch)
        
        #train the model
        if model.training:
            model.zero_grad()
            elbo.backward()
            optimizer.step()
        
        #
        average_epoch_elbo += elbo.item()/data_size

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def sample_n_save(model, epoch, path_extra = ""):
    with torch.no_grad():
        num_row = 5
        samples, im_means = model.sample(num_row * num_row)
        samples = make_grid(samples, nrow = num_row)
        path = "./samples" + path_extra +"/"
        create_dir(path)
        name = f"sample_{epoch}.png"
        plt.imsave(path + name, samples.cpu().numpy().transpose(1,2,0))
        

def manifold_plot_n_save(model):
    with torch.no_grad():
        num_row = 20
        grid = torch.linspace(0, 1, num_row)
        samples = [torch.erfinv(2 * torch.tensor([x, y]) - 1) * np.sqrt(2) for x in grid for y in grid]
        samples = torch.stack(samples).to(model.device)
        manifold = model.decoder(samples).view(-1, 1, 28, 28)
        image = make_grid(manifold, nrow = num_row)
        plt.imsave("manifold.png", image.cpu().numpy().transpose(1,2, 0))

def print_args(args):
  """
  Prints all entries of the config.
  """
  for key, value in vars(args).items():
    print(key + ' : ' + str(value))
    
def main(ARGS):
    data = bmnist()[:2]  # ignore test split
    
    #get device
    device = get_device(ARGS)
    
    print_args(ARGS)
    
    #initialize model
    model = VAE(z_dim=ARGS.zdim, device = device)
    print(f"Model loaded: \n{model}")
    
    #initialize optimizer
    optimizer = torch.optim.Adam(model.parameters())
    print(f"optimizer loaded: \n{optimizer}")
    
    path_model = "./VAE" + ARGS.experiment + "/"
    create_dir(path_model)
    sample_n_save(model, epoch = 0, path_extra = ARGS.experiment)
    print(f"first sample saved in {path_model}")
    
    train_curve, dev_curve = [], []
    
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, dev_elbo = elbos
        train_curve.append(train_elbo)
        dev_curve.append(dev_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} dev_elbo: {dev_elbo}")
        
        #save status after one epoch of training.
        sample_n_save(model, epoch = epoch, path_extra = ARGS.experiment)
        np.save(path_model+"train_elbo", train_curve)
        np.save(path_model+"dev_curve", dev_curve)
        torch.save(model.state_dict(), 
                   path_model + model.__class__.__name__ + ".pt")
        
    if ARGS.zdim == 2:
        manifold_plot_n_save(model)

    save_elbo_plot(train_curve, dev_curve, 'elbo.pdf')
    
def create_dir(path):
    os.makedirs(path, exist_ok=True)

def get_device(ARGS):
    #defines the device to be used.
    wanted_device = ARGS.device.lower()
    if wanted_device == 'cuda':
        #check if cuda is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        #cpu is the standard option
        device = torch.device('cpu')
        
    return device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=2, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--device', type = str, default='cpu',
                        help='torch device, "cpu" or "cuda"')
    parser.add_argument('--experiment', type = str, default='_2d',
                        help='experiment name')

    ARGS = parser.parse_args()

    main(ARGS)
