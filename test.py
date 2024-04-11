import argparse
import torch
import torch.nn as nn
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from unet import UNet
from hednet import HedNet
from hednet import EnsembleSkeletonNet

#from trainer import Trainer
from trainer_ofda import Trainer
from training_functions import get_device

def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet and SkeletonNet')
    parser.add_argument('-a', '--architecture', type=str, choices=["unet","skeleton"], default="unet", help='Architecture UNet or SkeletonNet')
    parser.add_argument('-l', '--loss', type=str, choices=["mae","mse",'smooth'], default="mae", help='Train Loss function')
    parser.add_argument('-nf', '--n_features', type=int, choices=[16,32,64], default=32, help='UNet 1st convolution features')
    parser.add_argument('-lri', '--lr_i', type=int, choices=[2,3,4,5,6], default=3, help='Learning Rate 10**i')
    parser.add_argument('-wdi', '--wd_i', type=int, choices=[3,4,5,6], default=6, help='Loss function')
    parser.add_argument('-e', '--ensemble_type', type=str, choices=["inner","outer"], default="inner", help='Ensemble type')
    parser.add_argument('-ts', '--testing_subset', type=str, choices=["synthetic","ofda"], default="synthetic", help='testing subset')
    return parser.parse_args()

if __name__=='__main__':
    
    args = get_args()
    
    print('Test Hyperparameters:')
    print('-'*20)
    print('Architecture:', args.architecture)
    print('Train Loss:', args.loss)
    print('Testing Subset:', args.testing_subset)
    
    #if args.architecture == 'unet':
    #    net = UNet(n_channels=3, n_classes=1, bilinear=False, n_features=args.n_features)
    #elif args.architecture == 'skeleton':
    #    net = HedNet(n_channels=3, n_classes=1, bilinear=False, side=4, n_features=32)
        
    #checkpoint = f"checkpoints/model_{args.architecture}_{args.loss}.pth"
    #checkpoint = f"checkpoints/model_{args.architecture}_{args.loss}_{args.n_features}_{args.lr_i}_{args.wd_i}.pth"
    
    if args.architecture == 'unet':
        net = UNet(n_channels=3, n_classes=1, bilinear=False, n_features=args.n_features)
        checkpoint = f'checkpoints/model3_unet_{args.loss}_{args.n_features}_{args.lr_i}_{args.wd_i}.pth'
        
    elif args.architecture == 'skeleton':
        if args.ensemble_type == 'inner':
            net = HedNet(n_channels=3, n_classes=1, bilinear=False, n_features=args.n_features, use_cuda=1)
        if args.ensemble_type == 'outer':
            model1 = HedNet(n_channels=3, n_classes=1, bilinear=False, side=0, n_features=32)
            model2 = HedNet(n_channels=3, n_classes=1, bilinear=False, side=4, n_features=32)
            net = EnsembleSkeletonNet(model1, model2)
        checkpoint = f"checkpoints/model3_snet_{args.ensemble_type}_{args.loss}.pth"
        
        
    device = get_device()
    print(f'Using {device} as device')
    net.to(device=device)
    
    trainer = Trainer(net, device, test_ofda_subset=(args.testing_subset == "ofda"))

    #img_path = "/home/aalejo/proyectos/dmnet/datasets/synthetic/train2/images/"
    #gt_path = "/home/aalejo/proyectos/dmnet/datasets/synthetic/train2/masks/"
    #trainer.load_test_dataset(img_path, gt_path)    
    
    trainer.net.load_state_dict(torch.load(checkpoint, map_location=device))

    mae, mse = trainer.test(batch_size=1, printlog=True)

    print('MAE:', mae)
    print('MSE:', mse)
    
    # guardar resultado
    #if args.testing_subset == 'synthetic':
    #    with open("resultados/test_log.csv",'a') as file_log:
    #        file_log.write(f'{args.architecture},{args.loss},{args.n_features},{args.lr_i},{args.wd_i},{mae},{mse}\n')
    if args.testing_subset == "ofda":
        #with open("resultados/testing_ofda.csv",'a') as file_log:
        with open("resultados/testing2_ofda.csv",'a') as file_log:
            file_log.write(f'{args.architecture},{checkpoint},{args.loss},{mae},{mse}\n')