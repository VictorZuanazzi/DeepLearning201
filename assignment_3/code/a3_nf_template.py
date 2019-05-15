import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from datasets.mnist import mnist
import os
from torchvision.utils import make_grid


def log_prior(x):
    """
    Compute the elementwise log probability of a standard Gaussian, i.e.
    N(x | mu=0, sigma=1).
    """
    two_pi = torch.tensor(2 * np.pi)  # for redability
    logp = torch.sum(- 0.5 * x.pow(2) - torch.log(torch.sqrt(two_pi)),
                     dim=1)

    return logp


def sample_prior(size, device):
    """
    Sample from a standard Gaussian.
    """

    # standard Gaussian mean and std
    mean = torch.zeros(size)
    std = torch.ones(size)

    # sample
    sample = torch.normal(mean, std).to(device)

    return sample


def get_mask():
    """mask stuff"""

    mask = np.zeros((28, 28), dtype='float32')
    for i in range(28):
        for j in range(28):
            if (i + j) % 2 == 0:
                mask[i, j] = 1

    mask = mask.reshape(1, 28 * 28)
    mask = torch.from_numpy(mask)

    return mask


class Coupling(torch.nn.Module):
    def __init__(self, c_in, mask, n_hidden=1024):
        super().__init__()
        self.n_hidden = n_hidden

        # Assigns mask to self.mask and creates reference for pytorch.
        self.register_buffer('mask', mask)

        # Create shared architecture to generate both the translation and
        # scale variables.
        # Suggestion: Linear ReLU Linear ReLU Linear.
        self.shared_net = torch.nn.Sequential(nn.Linear(c_in, n_hidden),
                                              nn.ReLU(),
                                              nn.Linear(n_hidden, n_hidden),
                                              nn.ReLU())

        self.t_net = nn.Linear(n_hidden, c_in)

        self.scale_net = torch.nn.Sequential(nn.Linear(n_hidden, c_in),
                                             nn.Tanh())

        # The nn should be initialized such that the weights of the last layer
        # is zero, so that its initial transform is identity.
        self.t_net.weight.data.zero_()
        self.t_net.bias.data.zero_()
        self.scale_net[0].weight.data.zero_()
        self.scale_net[0].bias.data.zero_()

    def forward(self, z, ldj, reverse=False):
        # Implement the forward and inverse for an affine coupling layer. Split
        # the input using the mask in self.mask. Transform one part with
        # Make sure to account for the log Jacobian determinant (ldj).
        # For reference, check: Density estimation using RealNVP.

        # NOTE: For stability, it is advised to model the scale via:
        # log_scale = tanh(h), where h is the scale-output
        # from the NN.

        # use mask on z
        z_mask = self.mask * z

        # Segue o FLOW
        hidden = self.shared_net(z_mask)
        t = self.t_net(hidden)
        scale = self.scale_net(hidden)

        if not reverse:
            # direct direction
            z = z_mask + (1 - self.mask) * (z * torch.exp(scale) + t)

            ldj += torch.sum((1 - self.mask) * scale,
                             dim=1)

        else:
            # reverse direction
            z = z_mask + (1 - self.mask) * (z - t) * torch.exp(-scale)

            # set the log determinant to zero for consistent output.
            ldj = torch.zeros_like(ldj)

        return z, ldj


class Flow(nn.Module):
    def __init__(self, shape, n_flows=4, device='cpu'):
        super().__init__()
        channels, = shape

        mask = get_mask().to(device)

        self.layers = torch.nn.ModuleList()

        for _ in range(n_flows):
            self.layers.append(Coupling(c_in=channels, mask=mask))
            self.layers.append(Coupling(c_in=channels, mask=1 - mask))

        self.z_shape = (channels,)

    def forward(self, z, logdet, reverse=False):
        if not reverse:
            for layer in self.layers:
                z, logdet = layer(z, logdet)
        else:
            for layer in reversed(self.layers):
                z, logdet = layer(z, logdet, reverse=True)

        return z, logdet


class Model(nn.Module):
    def __init__(self, shape, device='cpu'):
        super().__init__()
        self.flow = Flow(shape, device=device).to(device)
        self.device = device

    def dequantize(self, z):
        return z + torch.rand_like(z)

    def logit_normalize(self, z, logdet, reverse=False):
        """
        Inverse sigmoid normalization.
        """
        alpha = 1e-5

        if not reverse:
            # Divide by 256 and update ldj.
            z = z / 256.
            logdet -= np.log(256) * np.prod(z.size()[1:])

            # Logit normalize
            z = z * (1 - alpha) + alpha * 0.5
            logdet += torch.sum(-torch.log(z) - torch.log(1 - z), dim=1)
            z = torch.log(z) - torch.log(1 - z)

        else:
            # Inverse normalize
            logdet += torch.sum(torch.log(z) + torch.log(1 - z), dim=1)
            z = torch.sigmoid(z)

            # Multiply by 256.
            z = z * 256.
            logdet += np.log(256) * np.prod(z.size()[1:])

        return z, logdet

    def forward(self, input):
        """
        Given input, encode the input to z space. Also keep track of ldj.
        """
        z = input
        ldj = torch.zeros(z.size(0), device=z.device)

        z = self.dequantize(z)
        z, ldj = self.logit_normalize(z, ldj)

        z, ldj = self.flow(z, ldj)

        # Compute log_pz and log_px per example
        log_pz = log_prior(z)
        log_px = log_pz + ldj

        return log_px

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Sample from prior and create ldj.
        Then invert the flow and invert the logit_normalize.
        """

        z = sample_prior((n_samples,) + self.flow.z_shape, self.device)
        ldj = torch.zeros(z.size(0), device=z.device)

        # invert the flow
        z, ldj = self.flow.forward(z, ldj, reverse=True)
        z, ldj = self.logit_normalize(z, ldj, reverse=True)

        return z.to(self.device)


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average bpd ("bits per dimension" which is the negative
    log_2 likelihood per dimension) averaged over the complete epoch.
    """

    bpds = 0.0

    for i, (X, _) in enumerate(data):

        # forward pass
        log_px = model.forward(X.to(model.device))

        # calculate the loss
        loss = - log_px.mean()

        if model.training:
            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # things break from time to time, that keeps things from breaking!
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=5.0)
            optimizer.step()

        bpds += loss.item()

    # average bpds
    # log(2) converts it to log_2(x)
    avg_bpd = bpds / ((i + 1) * X.shape[1] * np.log(2))

    return avg_bpd


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average bpd for each.
    """
    traindata, valdata = data

    model.train()
    train_bpd = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_bpd = epoch_iter(model, valdata, optimizer)

    return train_bpd, val_bpd


def save_bpd_plot(train_curve, dev_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train bpd')
    plt.plot(dev_curve, label='validation bpd')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('bpd')
    plt.tight_layout()
    plt.savefig(filename)


def get_device(args):
    # defines the device to be used.
    wanted_device = args.device.lower()
    if wanted_device == 'cuda':
        # check if cuda is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        # cpu is the standard option
        device = torch.device('cpu')

    return device


def print_args(args):
    """
    Prints all entries of the config.
    """
    for key, value in vars(args).items():
        print(key + ' : ' + str(value))


def save_images(model, epoch, path, num_examples=25):
    generated = model.sample(num_examples).detach().reshape(num_examples, 1, 28, 28)
    grid = make_grid(generated, nrow=5, normalize=True).permute(1, 2, 0)
    name = f"epoch {epoch}.png"
    plt.imsave(path + name, grid)


def main(ARGS):
    # load data:
    data = mnist()[:2]  # ignore test split

    # load device
    device = get_device(ARGS)

    # print args:
    print_args(ARGS)
    print(f"Device used: {device}")

    # load model
    model = Model(shape=[784]).to(device)

    # load optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=ARGS.lr)

    # make directory to save images
    os.makedirs(ARGS.save_path + 'images_nfs',
                exist_ok=True)

    train_curve, dev_curve = [], []
    for epoch in range(ARGS.epochs):
        path = ARGS.save_path + 'images_nfs' + "/"
        save_images(model, epoch, path)

        train_bpd, val_bpd = run_epoch(model, data, optimizer)
        train_curve.append(train_bpd)
        dev_curve.append(val_bpd)
        print("[Epoch {epoch}] train bpd: {train_bpd:.5f} val_bpd: {val_bpd:.5f}".format(
            epoch=epoch, train_bpd=train_bpd, val_bpd=val_bpd))

        # save stats
        stats = {"train loss": train_curve, "dev_loss": dev_curve}
        np.save(ARGS.save_path + "stats.npy", stats)

        # save model
        torch.save(model.state_dict(),
                   ARGS.save_path + model.__class__.__name__ + ".pt")

    save_bpd_plot(train_curve,
                  dev_curve,
                  ARGS.save_path + 'nfs_bpd.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--device', default='cpu', type = str,
                        help='torch device intended, "cpu" or "cuda".')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--save_path', default='./nf2/', type=str,
                        help='define path where to save all subfolders')

    ARGS = parser.parse_args()

    main(ARGS)
