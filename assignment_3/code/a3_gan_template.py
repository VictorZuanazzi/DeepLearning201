import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import numpy as np


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-Linearity
        
        leak = 0.2
        
        #suggested implementation
        self.generator = nn.Sequential(nn.Linear(args.latent_dim, 128),
                                       nn.LeakyReLU(leak),
                                       nn.Linear(128,256),
                                       nn.BatchNorm1d(256),
                                       nn.LeakyReLU(leak),
                                       nn.Linear(256, 512),
                                       nn.BatchNorm1d(512),
                                       nn.LeakyReLU(leak),
                                       nn.Linear(512,1024),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(leak),
                                       nn.Linear(1024,784),
                                       nn.BatchNorm1d(784),
                                       nn.Tanh())
        

    def forward(self, z):
        
        # Generate images from z
        fake = self.generator(z)
        
        return fake


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-Linearity
        
        leak = 0.2
        
        #suggested implementation
        self.discriminator = nn.Sequential(nn.Linear(784, 512),
                                           nn.LeakyReLU(leak),
                                           nn.Linear(512, 256),
                                           nn.LeakyReLU(leak),
                                           nn.Linear(256, 1),
                                           nn.Sigmoid())
        

    def forward(self, img):
        
        # return discriminator score for img
        out = self.discriminator(img)
        return out

def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()


def get_device(args):
    #defines the device to be used.
    wanted_device = args.device.lower()
    if wanted_device == 'cuda':
        #check if cuda is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        #cpu is the standard option
        device = torch.device('cpu')
        
    return device


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    
    #device to be used
    device = get_device(args)
    print("device:", device)
    
    #put stuff in the device
    discriminator = discriminator.to(device)
    generator = generator.to(device)
    
    #a dict that keeps all the statistics of the training.
    stats = {}
    
    #losses
    stats["loss_G"] = []
    stats["loss_D"] = []
    
    #performance of the discriminator
    stats["accuracy"] = [0]  
    stats["precision"] = [0]
    stats["recall"] = [0]
    
    #performance of the generator
    stats["score"] = [0]
    
    #for readability
    latent_dim = args.latent_dim
    max_acc = args.max_acc
    freeze_D = args.freeze_D
    reset_D = args.reset_D
    path_images = args.save_images +"/"
    path_model = args.save_model + "/"
    epsilon = 1e-8
    
    for epoch in range(args.n_epochs):
        
        #keep stats of each step
        acc = [0]
        pre = [0]
        rec = [0]
        scr = [0]
        
        for i, (imgs, _) in enumerate(dataloader):

            x = imgs.view(-1, 784).to(device)
            
            current_batch = x.shape[0]

            # Train Generator
            z = torch.randn(current_batch, latent_dim, device = device)
            l_G = -torch.log(discriminator(generator(z))).sum()
            l_G.clamp(epsilon)
            
            optimizer_G.zero_grad()
            l_G.backward()
            optimizer_G.step()
            
            # Train Discriminator
            z = torch.randn(current_batch, latent_dim, device = device)
            
            #get classification of the discriminator
            real = discriminator(x)
            fake = discriminator(generator(z))
            
            l_D = -(torch.log(real) + torch.log(1 - fake)).sum()
            l_D.clamp(epsilon)
            
            if not((acc[-1] > max_acc) & freeze_D):
                #don't train the discriminator if it's accuracy is too high.
                #That is to give the generator a chance!
                optimizer_D.zero_grad()
                l_D.backward()
                optimizer_D.step()
            
            stats["loss_G"].append(l_G.item())
            stats["loss_D"].append(l_D.item())
            
            #stats time!
            TP = (real >= .5).sum().item() #True Positives
            TN = (fake < .5).sum().item() #True Negatives
            FP = (fake >= .5).sum().item() #False Positives
            FN = (real < .5).sum().item() #False Negatives

            acc.append((TP + TN) / (2.0 * current_batch)) #accuracy
            pre.append(TP / max((TP + FP), epsilon)) #precision
            rec.append(TP / max((TP + FN), epsilon)) #recall
            scr.append(FP / current_batch) #generator's score
      
        #Do some administration before starting next batch:      
               
        #print a status update
        print(f"################### Epoch: {epoch} ###################")
        print(f"Generator: {stats['loss_G'][-1]:0.5f},"
              f"score: {stats['score'][-1]:0.5f}")
        print(f"Discriminator Accuracy: {stats['accuracy'][-1]:0.2f},"
              f"Precision: {stats['precision'][-1]:0.2f},"
              f"Recall: {stats['recall'][-1]:0.2f},"
              f"Loss: {stats['loss_D'][-1]:0.5f}") 
        print(f"Last batch: TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
                
        #get stats for the last epoch
        stats["accuracy"].append(np.array(acc).mean())
        stats["precision"].append(np.array(pre).mean())
        stats["accuracy"].append(np.array(acc).mean())
        stats["recall"].append(np.array(rec).mean())
        stats["score"].append(np.array(scr).mean())
        
        #in case the Discriminator is too good, we just restart it!
        if reset_D & (stats["accuracy"][-1] > max_acc):
            print(f"Restarting the discriminator, acc: {stats['accuracy'][-1]:0.5f}")
            discriminator.apply(weight_reset)
            
        
        #save intermediary results
        np.save(path_model+"stats", stats)
    
        #save current state of the model
        best_G = ""
        if stats["score"][-1] == max(stats["score"]):
            best_G = "_best"
            
        best_D = ""
        if stats["accuracy"][-1] == max(stats["accuracy"]):
            best_D = "_best"
        
        torch.save(generator.state_dict(), 
                   path_model + generator.__class__.__name__ + best_G + ".pt")
        torch.save(discriminator.state_dict(), 
                   path_model + discriminator.__class__.__name__ + best_D +".pt")
        
        #save generated images
        save_image(generator(z[:25]).view(-1, 1, 28, 28),
                    path_images + 'fake_{}.png'.format(epoch),
                    nrow=5, 
                    normalize=True)
        
        #save a real batcch
        save_image(x[:25].view(-1, 1, 28, 28),
                    path_images + 'real_{}.png'.format(epoch),
                    nrow=5,
                    normalize=True)
    
def print_args(args):
  """
  Prints all entries of the config.
  """
  for key, value in vars(args).items():
    print(key + ' : ' + str(value))

def create_dir(folder, dataset):
    name = folder + '_' + dataset
    os.makedirs(name, exist_ok=True)
    return name
    
def load_model(model, folder_name, model_name):
    path = "./" + folder_name + "/"
    try:
        model.load_state_dict(torch.load(path + model_name))
    except:
        print(f"{model_name} not found, random initialization will be used.")
    
    return model

def main():
    # Create output directories
    args.save_images = create_dir(args.save_images, args.dataset)
    args.save_model = create_dir(args.save_model, args.dataset)
    
    #print the parameters
    print_args(args)
    
    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models
    generator = Generator()
    discriminator = Discriminator()
    
    #loads pre-trained models
    if args.continue_G:
        g_name = "Generator_best.pt"
        generator = load_model(generator, args.save_model, g_name)
        
    if args.continue_D:
        d_name = "Discriminator_best.pt"
        discriminator = load_model(discriminator, args.save_model, d_name)
    
    
    #initialize optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    print("All set to start the game!")
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    #define folders
    parser.add_argument('--save_images', type = str, default = 'images',
                        help='folder to save generated images.')
    parser.add_argument('--save_model', type = str, default = 'model',
                        help='folder to save the models.')
    parser.add_argument('--dataset', type = str, default = 'MNIST',
                        help='dataset to be used for training.')
    
    #training hyper parameters
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--device', type = str, default = 'cpu', 
                        help='torch device, "cpu" or "cuda"')
    parser.add_argument('--max_acc', type = float, default = 0.75,
                        help='accuracy to apply regularization to the discriminator')
    parser.add_argument('--freeze_D', type = bool, default = True,
                        help='If true does not train the discriminator when it reaches accuracy above args.max_acc')
    parser.add_argument('--reset_D', type = bool, default = False,
                        help='If True, the weights of the discriminator are reinitialized when the accuracy is > than args.max_acc')
    parser.add_argument('--continue_G', type = bool, default = True,
                        help='If True it continues training the previous best generator')
    parser.add_argument('--continue_D', type = bool, default = True,
                        help='If True it continues training the previous best discriminator')
    args = parser.parse_args()

    main()
