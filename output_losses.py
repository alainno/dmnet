import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
from torchvision.utils import make_grid
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from unet import UNet
from hednet import HedNet
from hednet import EnsembleSkeletonNet

from trainer import Trainer
from training_functions import get_device

def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet and SkeletonNet')
    parser.add_argument('-a', '--architecture', type=str, choices=["unet","skeleton"], default="unet", help='Architecture UNet or SkeletonNet')
    #parser.add_argument('-l', '--loss', type=str, choices=["mae","mse",'smooth'], default="mae", help='Train Loss function')
    parser.add_argument('-nf', '--n_features', type=int, choices=[16,32,64], default=32, help='UNet 1st convolution features')
    parser.add_argument('-lri', '--lr_i', type=int, choices=[2,3,4,5,6], default=3, help='Learning Rate 10**i')
    parser.add_argument('-wdi', '--wd_i', type=int, choices=[3,4,5,6], default=6, help='Loss function')
    parser.add_argument('-e', '--ensemble_type', type=str, choices=["inner","outer"], default="inner", help='Ensemble type')
    parser.add_argument('-ts', '--testing_subset', type=str, choices=["synthetic","ofda"], default="synthetic", help='testing subset')
    return parser.parse_args()

def plot_col(gs_index, plt, title, tensors):
    image = make_grid(tensors, nrow=1, padding=20, normalize=True, pad_value=1)
    plt.subplot(gs_index), plt.axis('off'), plt.title(title), plt.imshow(image.permute(1,2,0))


if __name__=='__main__':
    
    args = get_args()
    
    print('Test Hyperparameters:')
    print('-'*20)
    print('Architecture:', args.architecture)
    #print('Train Loss:', args.loss)
    print('Testing Subset:', args.testing_subset)
    
    losses = ['mae','mse','smooth']
    
    if args.architecture == 'unet':
        net = UNet(n_channels=3, n_classes=1, bilinear=False, n_features=args.n_features)
        #checkpoint = f'checkpoints/model3_unet_{args.loss}_{args.n_features}_{args.lr_i}_{args.wd_i}.pth'
        checkpoints = dict(map(lambda loss : (loss,f'checkpoints/model3_unet_{loss}_{args.n_features}_{args.lr_i}_{args.wd_i}.pth'), losses))
    elif args.architecture == 'skeleton':
        if args.ensemble_type == 'inner':
            net = HedNet(n_channels=3, n_classes=1, bilinear=False, n_features=args.n_features, use_cuda=1)
        if args.ensemble_type == 'outer':
            model1 = HedNet(n_channels=3, n_classes=1, bilinear=False, side=0, n_features=32)
            model2 = HedNet(n_channels=3, n_classes=1, bilinear=False, side=4, n_features=32)
            net = EnsembleSkeletonNet(model1, model2)
        #checkpoint = f"checkpoints/model3_snet_{args.ensemble_type}_{args.loss}.pth"
        checkpoints = dict(map(lambda loss : (loss,f"checkpoints/model3_snet_{args.ensemble_type}_{loss}.pth"), losses))
        
        
    device = get_device()
    print(f'Using {device} as device')
    net.to(device=device)
    
    trainer = Trainer(net, device, test_ofda_subset=(args.testing_subset == "ofda"))

    
    ################################
    plt.figure(figsize=(5*2,4*2))
    
    ###########
    # grid
    gs = gridspec.GridSpec(1, 5)
    gs.update(wspace=0, hspace=0.1)
    
    label_loss = {'mae':'L1 Loss','mse':'L2 Loss','smooth':'Smooth L1 Loss'}
    
    batch = next(iter(trainer.test_data_loader))
    
    for count,loss in enumerate(losses):
        trainer.net.load_state_dict(torch.load(checkpoints[loss], map_location=device))
        inputs, gt, output = trainer.test_output(batch_size=4, batch=batch)
        
        if count==0:
            plot_col(gs[count], plt, 'Image', inputs)
            plot_col(gs[count+1], plt, 'Ground Truth', gt)
            
        plot_col(gs[count+2], plt, f'{label_loss[loss]}', output)
    
    plt.savefig("output-losses.png", bbox_inches='tight', pad_inches=0)