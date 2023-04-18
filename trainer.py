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
                
                #print(output.shape)

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
                    loss = self.criterion(output, ground_truth)

                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)':loss.item()})
                pbar.update(input.shape[0]) # samples in batch

        return epoch_loss / len(self.val_data_loader)

    def train_and_validate(self, epochs, criterion, optimizer, scheduler, model_output_path, max_epochs_without_improve=10):
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
        max_epochs_without_improve = max_epochs_without_improve
        train_losses, val_losses = [],[]

        for epoch in range(epochs):
            #print(f"Epoch {epoch+1} de {epochs}")
            train_loss = self.__train(epoch)
            train_losses.append(train_loss)

            val_loss = self.__validate(epoch)
            val_losses.append(val_loss)
            
            #if (epoch+1) % 10 == 0:
            #    self.scheduler.step()
            self.scheduler.step() # para StepLR cada step_size

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
        #return min(train_losses),min(val_losses)
        print("Min Training Loss:", min(train_losses))
        print("Min Validation Loss:", min(val_losses))
        
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
            input = input.to(device=self.device, dtype=torch.float32)
            groundtruth = groundtruth.to(device=self.device, dtype=torch.float32)
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
