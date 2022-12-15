import argparse
import torch
import torch.nn as nn
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from unet import UNet
from hednet import HedNet

def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet and SkeletonNet')
    parser.add_argument('-a', '--architecture', type=str, choices=["unet","skeleton"], default="unet", help='Architecture UNet or SkeletonNet')
    parser.add_argument('-l', '--loss', type=str, choices=["mae","mse",'smooth'], default="mae", help='Train Loss function')
    return parser.parse_args()

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device

def get_test_loader(batch_size=2):
    test_img_path = "/home/aalejo/proyectos/dmnet/datasets/synthetic/test/images/"
    test_gt_path = "/home/aalejo/proyectos/dmnet/datasets/synthetic/test/masks/"

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.114, 0.114, 0.114],std=[0.237, 0.237, 0.237])
    ])

    test_dataset = BasicDataset(imgs_dir = test_img_path, masks_dir = test_gt_path, transforms=trans, mask_h5=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    # test_loader = [
    #     {'image':torch.rand(3,5),'mask':torch.rand(3,5)},
    #     {'image':torch.rand(3,5),'mask':torch.rand(3,5)},
    #     {'image':torch.rand(3,5),'mask':torch.rand(3,5)},
    #     {'image':torch.rand(3,5),'mask':torch.rand(3,5)},
    # ]
    return test_loader


def test(model, device, batch_size=2, printlog=False):
    if printlog:
        print("Iniciando el testing...")
    
    test_loader = get_test_loader(batch_size=batch_size)
    
    if printlog:
        print(f'{len(test_loader)} test batches of {batch_size} samples loaded')

    criterion = nn.L1Loss()
    criterion2 = nn.MSELoss()

    model.eval()

    test_loss, test_loss2 = 0,0
    maes, mses = 0,0
    for batch in test_loader:
        input, groundtruth = batch['image'],batch['mask']
        input = input.to(device=device, dtype=torch.float32)
        groundtruth = groundtruth.to(device=device, dtype=torch.float32)
        #ground_truh = np.asarray(ground_truth)
        #print(type(ground_truth))
        #print(ground_truth.shape)

        with torch.no_grad():
            output = model(input)
            
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

    test_loss /= len(test_loader)
    #return test_loss, maes/len(test_loader)
    #return test_loss, test_loss2 / len(test_loader)
    return test_loss, mses / len(test_loader)



if __name__=='__main__':
    
    args = get_args()
    
    print('Test Hyperparameters:')
    print('-'*20)
    print('Architecture:', args.architecture)
    print('Train Loss:', args.loss)
    
    if args.architecture == 'unet':
        net = UNet(n_channels=3, n_classes=1, bilinear=False, n_features=32)
    elif args.architecture == 'skeleton':
        net = HedNet(n_channels=3, n_classes=1, bilinear=False, side=4, n_features=32)
        
    checkpoint = f"checkpoints/model_{args.architecture}_{args.loss}.pth"
   
    device = get_device()
    net.to(device=device)
    net.load_state_dict(torch.load(checkpoint))

    mae, mse = test(net, device, batch_size=4, printlog=False)

    print('MAE:', mae)
    print('MSE:', mse)