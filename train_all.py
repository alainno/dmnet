import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import optim
import torch.nn as nn

from hednet import HedNet
from unet import UNet
from utils.dataset import BasicDataset
from net_utils import *

from tqdm import tqdm
import pandas as pd

import sys
import numpy as np
from datetime import datetime

def train_net(net, criterion, target='MODEL.pth'):
    """ funcion de entrenamiento """
    # hiperparametros:
    #criterion = nn.MSELoss().cuda()
    #criterion = nn.HuberLoss()
    #optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.96)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.96)

    # loss history:
    train_losses = []
    val_losses = []

    # early stopping vars:
    best_prec1 = np.Inf #1e6
    epochs_no_improve = 0
    n_epochs_stop = 10

    # train loop:
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Train Epoch {epoch+1}/{epochs}') as pbar:
            for batch in train_loader:
                imgs, true_masks = batch['image'], batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                masks_pred = net(imgs)
                
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item() * imgs.size(0)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                
        train_losses.append(epoch_loss/len(train))
        
        #validacion:
        net.eval()
        epoch_loss = 0
        
        with tqdm(total=n_val, desc=f'Val Epoch {epoch+1}/{epochs}') as pbar:
            for batch in val_loader:
                imgs, true_masks = batch['image'], batch['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    mask_pred = net(imgs)
                    loss = criterion(mask_pred, true_masks)
                
                epoch_loss += loss.item() * imgs.size(0)
                
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(imgs.shape[0])

        val_losses.append(epoch_loss/len(val))
        
        if (epoch+1) % 10 == 0:
            scheduler.step()

        # se guarda el modelo si es mejor que el anterior:
        prec1 = epoch_loss/n_val
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        
        if is_best:
            epochs_no_improve = 0
            torch.save(net.state_dict(), target) 
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!')
                break    

    print(f'The best Loss: {min(val_losses)}')    
    return best_prec1


if __name__ == "__main__":
    
    
    print(torch.__version__)
    #sys.exit(0)
    now = datetime.now()

    # definiendo los hiperparametros:
    epochs = 500
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    val_percent = 0.2
    img_scale = 1
    dir_img = 'data_dm_overlapping/imgs/'
    dir_mask = 'data_dm_overlapping/masks/'
    batch_size = 4
    lr = 0.0001
    weight_decay = 5*(10**-7)

    # definiendo el dataset:
    dataset = BasicDataset(dir_img, dir_mask, img_scale, transforms=transforms.Compose([
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize(mean=[0.114, 0.114, 0.114],std=[0.237, 0.237, 0.237])
                                                                    ]), mask_h5=True)

    # definimos los conjunto de entrenamiento, validacion y pruebas:
    n_val = int(len(dataset) * val_percent)
    #n_test = n_val
    n_train = len(dataset) - n_val# - n_test
    #train, val, test = random_split(dataset, [n_train, n_val, n_test])
    train, val = random_split(dataset, [n_train, n_val])

    # batch loader:
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    #test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # Training:
    model_list = []
    error_list = []
    # for test:
    #lote = next(iter(test_loader))
    #img_path_list = lote['path']
    
    
    criterion = nn.MSELoss()
    '''
    print('Training U-Net (MSE)...')
    model_list.append('U-Net (MSE)')
    
    net = UNet(n_channels=3, n_classes=1, bilinear=False, n_features=32)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device=device)      

    error = train_net(net, criterion, 'checkpoints/MODEL_unet_mse.pth')
    error_list.append(error)
    '''
    print('Training U-Net (MSE)...')
    model_list.append('U-Net (MSE)')
    
    net = UNet(n_channels=3, n_classes=1, bilinear=False, n_features=64)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device=device)      

    error = train_net(net, criterion, 'checkpoints/MODEL_unet_mse.pth')
    error_list.append(error)    
    
    print('Training SkeletonNet (MSE)...')
    model_list.append('SkeletonNet (MSE)')
    
    
    net = HedNet(n_channels=3, n_classes=1, bilinear=False, side=4)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)    
    net.to(device=device)
    
    error = train_net(net, criterion, 'checkpoints/MODEL_snet_mse.pth')
    error_list.append(error)
    
    ########
    criterion = nn.SmoothL1Loss()

    print('Training U-Net (SmoothL1Loss)...')
    model_list.append('U-Net (SmoothL1Loss)')
    
    net = UNet(n_channels=3, n_classes=1, bilinear=False, n_features=64)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device=device)      

    error = train_net(net, criterion, 'checkpoints/MODEL_unet_sl1.pth')
    error_list.append(error)
    
    # test U-Net
    #net.load_state_dict(torch.load('MODEL.pth'))
    #net.eval()
    #diameter_means_unet = get_diameters(net, img_path_list)
    
    print('Training SkeletonNet (SmoothL1Loss)...')
    model_list.append('SkeletonNet (SmoothL1Loss)')
    
    net = HedNet(n_channels=3, n_classes=1, bilinear=False, side=4)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)    
    net.to(device=device)
    
    error = train_net(net, criterion, 'checkpoints/MODEL_snet_sl1.pth')
    error_list.append(error)
    
    # test Skeleton
    #net.load_state_dict(torch.load('MODEL.pth'))
    #net.eval()
    #diameter_means_snet = get_diameters(net, img_path_list)
        
    '''
    model_list = []
    mse_list = []
    
    lrs = [10**-i for i in range(2,7)]
    weight_decays = [5*(10**-i) for i in range(3,7)]
    
    for lr in lrs:
        for weight_decay in weight_decays:
            
            model_list.append("UNet(lr=" + str(lr) + ",decay=" + str(weight_decay))

            net = UNet(n_channels=3, n_classes=1, bilinear=False)
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                net = nn.DataParallel(net)

            net.to(device=device)

            mse = train_net(net, lr, weight_decay)
            mse_list.append(mse)

            model_list.append("SkeletonNet(lr=" + str(lr) + ",decay=" + str(weight_decay))
            #for i in range(5):
            net = HedNet(n_channels=3, n_classes=1, bilinear=False, side=4)
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                net = nn.DataParallel(net)    
            net.to(device=device)

            mse = train_net(net, lr, weight_decay)
            mse_list.append(mse)
            
            
    '''
    dataset = {'Modelo':model_list,#,'SNet 1','SNet 2','SNet 3','SNet 4','SNet 5'],
              'Loss':error_list}
    df = pd.DataFrame(dataset)
    df.to_csv('resultados/losses_'+now.strftime("%Y%m%d%H%M")+'.csv', index=False)
    
    
    # realizamos las predicciones y el procesamiento posterior
    '''
    gts_path = "data_dm_overlapping/diameter_means.pkl"
    
    mydataset = {
        'Imagen': img_path_list,
        'Ground Truth': matchGt(img_path_list, gts_path),
        'U-Net': diameter_means_unet,
        'SkeletonNet': diameter_means_snet
    }

    df = pd.DataFrame(mydataset)
    #print(tabulate(df, headers = 'keys', tablefmt = 'psql'))
    df.to_csv('diametros_'+now.strftime("%Y%m%d%H%M")+'.csv', index=False)
    '''