# import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import argparse

from unet import UNet
from hednet import HedNet
from utils.dataset import BasicDataset

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet and SkeletonNet on images and target masks')
    parser.add_argument('-a', '--architecture', type=str, choices=["unet","skeleton"], default="unet", help='Architecture UNet or SkeletonNet')
    parser.add_argument('-l', '--loss', type=str, choices=["mae","mse",'smooth'], default="mae", help='Loss function')
    parser.add_argument('-nf', '--n_features', type=int, choices=[16,32,64], default=32, help='UNet 1st convolution features')
    parser.add_argument('-lri', '--lr_i', type=int, choices=[2,3,4,5,6], default=3, help='Learning Rate 10**i')
    parser.add_argument('-wdi', '--wd_i', type=int, choices=[3,4,5,6], default=6, help='Loss function')
    return parser.parse_args()

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device


class Trainer:

    def __init__(self, net, device):
        self.net = net
        self.device = device

        img_path = "/home/aalejo/proyectos/dmnet/datasets/synthetic/train/images/"
        gt_path = "/home/aalejo/proyectos/dmnet/datasets/synthetic/train/masks/"

        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.114, 0.114, 0.114],std=[0.237, 0.237, 0.237])
        ])

        dataset = BasicDataset(imgs_dir = img_path, masks_dir = gt_path, transforms=trans, mask_h5=True)

        val_percent = 0.2
        batch_size = 4

        self.n_val = int(len(dataset) * val_percent)
        self.n_train = len(dataset) - self.n_val
        train, val = random_split(dataset, [self.n_train, self.n_val])

        self.train_data_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=False)
        self.val_data_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

        # self.train_data_loader = self.__get_train_data_loader()
        # self.val_data_loader = self.__get_val_data_loader()
        
        
    def __init_test_dataset(self, batch_size=2):
        test_img_path = "/home/aalejo/proyectos/dmnet/datasets/synthetic/test/images/"
        test_gt_path = "/home/aalejo/proyectos/dmnet/datasets/synthetic/test/masks/"

        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.114, 0.114, 0.114],std=[0.237, 0.237, 0.237])
        ])

        test_dataset = BasicDataset(imgs_dir = test_img_path, masks_dir = test_gt_path, transforms=trans, mask_h5=True)
        self.test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        

    def __train(self, epoch):
        ''' Método de entrenamiento del modelo para una época'''
        self.net.train()
        epoch_loss = 0

        with tqdm(total=self.n_train, desc=f'Train Epoch {epoch+1}/{self.epochs}') as pbar:
            for batch in self.train_data_loader:
                input,ground_truth = batch['image'],batch['mask']
                input = input.to(device=self.device, dtype=torch.float32)
                ground_truth = ground_truth.to(device=self.device, dtype=torch.float32)
                
                output = self.net(input)

                loss = self.criterion(output, ground_truth)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(input.shape[0])
        
        return epoch_loss / len(self.train_data_loader)

    def __validate(self, epoch):
        ''' Método de validación del modelo'''
        self.net.eval()
        epoch_loss = 0

        with tqdm(total=self.n_val, desc=f'Val Epoch {epoch+1}/{self.epochs}') as pbar:
            for batch in self.val_data_loader:
                input,ground_truth = batch['image'],batch['mask']
                input = input.to(device=self.device, dtype=torch.float32)
                ground_truth = ground_truth.to(device=self.device, dtype=torch.float32)

                with torch.no_grad():
                    output = self.net(input)
                    loss = criterion(output, ground_truth)

                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)':loss.item()})
                pbar.update(input.shape[0]) # samples in batch

        return epoch_loss / len(self.val_data_loader)

    def train_and_validate(self, epochs, criterion, optimizer, scheduler, model_output_path):
        ''' Entrenar y validar modelo por varias épocas'''
        # train_data_loader = self.get_train_data_loader()
        # val_data_loader = self.get_val_data_loader()

        # criterion = torch.nn.L1Loss()
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        min_val_loss = np.Inf
        epochs_without_improve = 0
        max_epochs_without_improve = 15#10
        train_losses, val_losses = [],[]

        for epoch in range(epochs):
            print(f"Epoch {epoch+1} de {epochs}")
            train_loss = self.__train(epoch)
            train_losses.append(train_loss)

            val_loss = self.__validate(epoch)
            val_losses.append(val_loss)
            
            #if (epoch+1) % 10 == 0:
            #    self.scheduler.step()
            self.scheduler.step() # para StepLR cada step_size

            self.scheduler.step()

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                epochs_without_improve = 0
                torch.save(self.net.state_dict(), model_output_path)
            else:
                epochs_without_improve += 1
                if epochs_without_improve > max_epochs_without_improve:
                    print("Early stopping!")
                    break

        # print(min_val_loss)
        return min(train_losses),min(val_losses)
    
    def test(self, batch_size=4, printlog=False):
        if printlog:
            print("Iniciando el testing...")

        #test_loader = get_test_loader(batch_size=batch_size)
        self.__init_test_dataset(batch_size=batch_size)

        if printlog:
            print(f'{len(self.test_data_loader)} test batches of {batch_size} samples loaded')

        criterion = nn.L1Loss()
        criterion2 = nn.MSELoss()

        self.net.eval()

        test_loss, test_loss2 = 0, 0
        maes, mses = 0, 0
        for batch in self.test_data_loader:
            input, groundtruth = batch['image'], batch['mask']
            input = input.to(device=device, dtype=torch.float32)
            groundtruth = groundtruth.to(device=device, dtype=torch.float32)
            #ground_truh = np.asarray(ground_truth)
            #print(type(ground_truth))
            #print(ground_truth.shape)

            with torch.no_grad():
                output = self.net(input)

                #print(output.shape)
                loss = criterion(output, groundtruth)

                loss2 = criterion2(output, groundtruth)

                mae,mse = 0,0
                for k in range(output.shape[0]):
                    a = output.detach().cpu().numpy()[k].squeeze()
                    b = groundtruth.detach().cpu().numpy()[k].squeeze()
                    for i in range(a.shape[0]):
                        for j in range(a.shape[1]):
                            mae += abs(a[i][j]-b[i][j])
                            mse += abs(a[i][j]-b[i][j])**2
                    #mae += abs(output.detach().cpu().numpy()[i].squeeze().sum()-groundtruth.detach().cpu().numpy()[i].squeeze().sum())
                maes += mae/(input.shape[0]*256*256)
                mses += mse/(input.shape[0]*256*256)
                #print('mae=',)

            test_loss += loss.item()
            test_loss2 += loss2.item()
            #test_loss += mae / input.shape[0]

        test_loss /= len(self.test_data_loader)
        #return test_loss, maes/len(test_loader)
        #return test_loss, test_loss2 / len(test_loader)
        return test_loss, mses / len(self.test_data_loader)

        


    
    
if __name__ == "__main__":

    #print(torch.__version__)
    args = get_args()
    
    print('Training Hyperparameters:')
    print('-'*20)
    print('Architecture:', args.architecture)
    print('Loss Function:', args.loss)
    print('Number of Features:', args.n_features)
    print('Learning Rate:', args.lr_i)
    print('Weight Decay:', args.wd_i)
    
    if args.architecture == 'unet':
        net = UNet(n_channels=3, n_classes=1, bilinear=False, n_features=args.n_features)
        optimizer = torch.optim.Adam(net.parameters(), lr=10**-(args.lr_i), weight_decay=5*(10**-(args.wd_i)))
        
    elif args.architecture == 'skeleton':
        net = HedNet(n_channels=3, n_classes=1, bilinear=False, side=5, n_features=args.n_features)
        optimizer = torch.optim.Adam(net.parameters(), lr=10**-3)
        
    device = get_device()
    net.to(device=device)

    if args.loss == 'mae':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'mse':
        criterion = torch.nn.MSELoss()
    elif args.loss == 'smooth':
        criterion = torch.nn.SmoothL1Loss()
    
    #checkpoint = f"checkpoints/model_{args.architecture}_{args.loss}.pth"
    # criterion = torch.nn.MSELoss()
    
    #optimizer = torch.optim.Adam(net.parameters(), lr=10**-4, weight_decay=5*(10**-6))
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.96)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.96) # ExponentialLR cada 'step_size' pasos
    
    model_output_path = f"checkpoints/model_{args.architecture}_{args.loss}_{args.n_features}_{args.lr_i}_{args.wd_i}.pth"

    trainer = Trainer(net, device)
    min_train_loss, min_val_loss = trainer.train_and_validate(epochs=500,
                                                                criterion=criterion,
                                                                optimizer=optimizer,
                                                                scheduler=scheduler,
                                                                model_output_path=model_output_path)

    print("Min Training Loss:", min_train_loss)
    print("Min Validation Loss:", min_val_loss)
    
    ###########
    # Test
    trainer.net.load_state_dict(torch.load(model_output_path))
    mae, mse = trainer.test(batch_size=4, printlog=False)

    print('MAE:', mae)
    print('MSE:', mse)
    
    # guardar resultado
    with open("resultados/test_log.csv",'a') as file_log:
        file_log.write(f'{args.architecture},{args.loss},{args.n_features},{args.lr_i},{args.wd_i},{mae},{mse}\n')
    


