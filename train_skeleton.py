import torch
import torch.nn as nn
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import argparse
from pytorch_model_summary import summary

from unet import UNet
from hednet import HedNet
from hednet import EnsembleSkeletonNet

from training_functions import get_args, get_device
#from trainer import Trainer
from trainer_ofda import Trainer


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet and SkeletonNet on images and target masks')
    parser.add_argument('-e', '--ensemble_type', type=str, choices=["inner","outer"], default="inner", help='Ensemble type')
    parser.add_argument('-l', '--loss', type=str, choices=["mae","mse",'smooth'], default="mae", help='Loss function')
    parser.add_argument('-m', '--max_epochs_without_improve', type=int, choices=range(11,20), default=15, help='Early stopping')
    parser.add_argument('-ts', '--testing_subset', type=str, choices=["synthetic","ofda"], default="synthetic", help='testing subset')
    #parser.add_argument('-nf', '--n_features', type=int, choices=[16,32,64], default=32, help='UNet 1st convolution features')
    #parser.add_argument('-lri', '--lr_i', type=int, choices=[2,3,4,5,6], default=3, help='Learning Rate 10**i')
    #parser.add_argument('-wdi', '--wd_i', type=int, choices=[3,4,5,6], default=6, help='Loss function')
    return parser.parse_args()


if __name__ == '__main__':
    
    args = get_args()
    
    print("Training SkeletonNet Regression")
    print("-"*30)
    print("Ensemble type:", args.ensemble_type)
    print('Loss Function:', args.loss)
    print('Maximum of epochs without improve:', args.max_epochs_without_improve)
    
   
    
    #model1_output_path = "./checkpoints/model_ensemble_snet.pth"
    #model2_output_path = "./checkpoints/model_snet_mae_32_3_4.pth"
    #model1_output_path = f"./checkpoints/model_snet_{args.ensemble_type}_{args.loss}.pth"
    model1_output_path = f"./checkpoints/model3_snet_{args.ensemble_type}_{args.loss}.pth"
    
    if args.ensemble_type == 'outer':
        model1 = HedNet(n_channels=3, n_classes=1, bilinear=False, side=3, n_features=32, use_cuda=1)
        model2 = HedNet(n_channels=3, n_classes=1, bilinear=False, side=4, n_features=32, use_cuda=1)
        ensemble_model = EnsembleSkeletonNet(model1, model2)
    else:
        ensemble_model = HedNet(n_channels=3, n_classes=1, bilinear=False, n_features=32, use_cuda=1)
        
    
    if args.loss == 'mae':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'mse':
        criterion = torch.nn.MSELoss()
    elif args.loss == 'smooth':
        criterion = torch.nn.SmoothL1Loss() 
    
    
    device = get_device()
    ensemble_model.to(device=device)
    #model2.to(device=device)
    
    #x = torch.zeros((1, 3, 198, 189))
    #x = x.to(device=device, dtype=torch.float32)
    #print(summary(ensemble_model, x))
    
    
    optimizer1 = torch.optim.Adam(ensemble_model.parameters(), lr=10**-3)
    #optimizer2 = torch.optim.Adam(model2.parameters(), lr=10**-3)
        
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=10)
    #scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=10, gamma=0.96)
    
    trainer1 = Trainer(ensemble_model, device, test_ofda_subset=(args.testing_subset == "ofda"))
    
    #img_path = "/home/aalejo/proyectos/dmnet/datasets/synthetic/train2/images/"
    #gt_path = "/home/aalejo/proyectos/dmnet/datasets/synthetic/train2/masks/"
    #trainer1.load_dataset(img_path, gt_path)
    
    trainer1.train_and_validate(epochs=500,
                                criterion=criterion,
                                optimizer=optimizer1,
                                scheduler=scheduler1,
                                model_output_path=model1_output_path,
                                max_epochs_without_improve=args.max_epochs_without_improve)
    
    #trainer2 = Trainer(model2, device)  
    #trainer2.train_and_validate(epochs=500,
    #                            criterion=criterion,
    #                            optimizer=optimizer2,
    #                            scheduler=scheduler2,
    #                            model_output_path=model2_output_path)
    
    
    
    '''
    # Test
    #model1.load_state_dict(torch.load(model1_output_path))
    #model2.load_state_dict(torch.load(model2_output_path))
    trainer1.net.load_state_dict(torch.load(model1_output_path))
    
    #
    #mae, mse = ensemble_test(model1, model2, batch_size=4, printlog=True)
    mae, mse = trainer1.test(batch_size=4, printlog=False)
    
    print('MAE:', mae)
    print('MSE:', mse)
    
    # guardar resultado
    #with open("resultados/test_skeleton_log.csv",'a') as file_log:
    with open("resultados/test2_skeleton_log.csv",'a') as file_log:
        #file_log.write(f'{args.architecture},{args.loss},{args.n_features},{args.lr_i},{args.wd_i},{mae},{mse}\n')
        #file_log.write(f'{args.ensemble_type},{args.loss},{args.max_epochs_without_improve},{mae},{mse},{args.testing_subset}\n')
        file_log.write(f'{args.ensemble_type},{args.loss},{args.max_epochs_without_improve},{mae},{mse},{args.testing_subset}\n')
    '''