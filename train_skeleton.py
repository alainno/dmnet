import torch
import torch.nn as nn
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import argparse

from unet import UNet
from hednet import HedNet
from hednet import EnsembleSkeletonNet

from training_functions import get_args, get_device
from trainer import Trainer


def get_test_loader(batch_size=2):
    test_img_path = "/home/aalejo/proyectos/dmnet/datasets/synthetic/test/images/"
    test_gt_path = "/home/aalejo/proyectos/dmnet/datasets/synthetic/test/masks/"

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.114, 0.114, 0.114],std=[0.237, 0.237, 0.237])
    ])

    test_dataset = BasicDataset(imgs_dir = test_img_path, masks_dir = test_gt_path, transforms=trans, mask_h5=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    return test_loader

'''
def ensemble_test(model1, model2, batch_size=4, printlog=False):
    if printlog:
        print("Iniciando el testing...")

    test_loader = get_test_loader(batch_size=batch_size)

    if printlog:
        print(f'{len(test_loader)} test batches of {batch_size} samples loaded')

    criterion = nn.L1Loss()
    criterion2 = nn.MSELoss()

    model1.eval()
    model2.eval()

    test_loss, test_loss2 = 0, 0
    maes, mses = 0, 0
    
    for batch in test_loader:
        input, groundtruth = batch['image'], batch['mask']
        input = input.to(device=device, dtype=torch.float32)
        groundtruth = groundtruth.to(device=device, dtype=torch.float32)


        with torch.no_grad():
            output1 = model1(input)
            output2 = model2(input)
            
            output = (output1 + output2) / 2

            loss = criterion(output, groundtruth)

            loss2 = criterion2(output, groundtruth)

            mae, mse = 0, 0
            for k in range(output.shape[0]):
                a = output.detach().cpu().numpy()[k].squeeze()
                b = groundtruth.detach().cpu().numpy()[k].squeeze()
                for i in range(a.shape[0]):
                    for j in range(a.shape[1]):
                        mae += abs(a[i][j]-b[i][j])
                        mse += abs(a[i][j]-b[i][j])**2

            maes += mae/(input.shape[0]*256*256)
            mses += mse/(input.shape[0]*256*256)

        test_loss += loss.item()
        test_loss2 += loss2.item()

    if printlog:
        print(test_loss/len(test_loader), test_loss2/len(test_loader), maes/len(test_loader), mses/len(test_loader))

    return test_loss/len(test_loader), mses/len(test_loader)
'''

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet and SkeletonNet on images and target masks')
    parser.add_argument('-e', '--ensemble_type', type=str, choices=["inner","outer"], default="inner", help='Ensemble type')
    parser.add_argument('-l', '--loss', type=str, choices=["mae","mse",'smooth'], default="mae", help='Loss function')
    parser.add_argument('-m', '--max_epochs_without_improve', type=int, choices=[10,15,20], default=10, help='Early stopping')
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
    
    
    model1_output_path = "./checkpoints/model_ensemble_snet.pth"
    #model2_output_path = "./checkpoints/model_snet_mae_32_3_4.pth"
    
    if args.ensemble_type == 'outer':    
        model1 = HedNet(n_channels=3, n_classes=1, bilinear=False, side=0, n_features=32)
        model2 = HedNet(n_channels=3, n_classes=1, bilinear=False, side=4, n_features=32)
        ensemble_model = EnsembleSkeletonNet(model1, model2)
    else:
        ensemble_model = HedNet(n_channels=3, n_classes=1, bilinear=False, n_features=32)
    
    
    if args.loss == 'mae':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'mse':
        criterion = torch.nn.MSELoss()
    elif args.loss == 'smooth':
        criterion = torch.nn.SmoothL1Loss() 
    
    
    device = get_device()
    ensemble_model.to(device=device)
    #model2.to(device=device)
    
    optimizer1 = torch.optim.Adam(ensemble_model.parameters(), lr=10**-3)
    #optimizer2 = torch.optim.Adam(model2.parameters(), lr=10**-3)
        
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=10)
    #scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=10, gamma=0.96)
    
    trainer1 = Trainer(ensemble_model, device)
    
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
    with open("resultados/test_skeleton_log.csv",'a') as file_log:
        file_log.write(f'{args.ensemble_type},{args.loss},{args.max_epochs_without_improve},{mae},{mse}\n')
        #file_log.write(f'{args.architecture},{args.loss},{args.n_features},{args.lr_i},{args.wd_i},{mae},{mse}\n')