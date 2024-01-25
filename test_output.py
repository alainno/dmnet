# recibe un batch
# shown images result of predicted distance map 
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.utils import make_grid

from unet import UNet
from hednet import HedNet
from trainer import Trainer

def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet and SkeletonNet')
    parser.add_argument('-a', '--architecture', type=str, choices=["unet","skeleton"], default="unet", help='Architecture UNet or SkeletonNet')
    parser.add_argument('-l', '--loss', type=str, choices=["mae","mse",'smooth'], default="mae", help='Train Loss function')
    #parser.add_argument('-nf', '--n_features', type=int, choices=[16,32,64], default=32, help='UNet 1st convolution features')
    #parser.add_argument('-lri', '--lr_i', type=int, choices=[2,3,4,5,6], default=3, help='Learning Rate 10**i')
    #parser.add_argument('-wdi', '--wd_i', type=int, choices=[3,4,5,6], default=6, help='Loss function')
    parser.add_argument('-ts', '--testing_subset', type=str, choices=["synthetic","ofda"], default="synthetic", help='testing subset')
    return parser.parse_args()

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device


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
        if args.loss == 'mae':
            checkpoint = 'checkpoints/model_unet_mae_64_3_6.pth'
            net = UNet(n_channels=3, n_classes=1, bilinear=False, n_features=64)
        elif args.loss == 'mse':
            checkpoint = 'checkpoints/model_unet_mse_32_3_6.pth'
            net = UNet(n_channels=3, n_classes=1, bilinear=False, n_features=32)
        elif args.loss == 'smooth':
            checkpoint = 'checkpoints/model_unet_smooth_32_3_6.pth'
            net = UNet(n_channels=3, n_classes=1, bilinear=False, n_features=32)
    elif args.architecture == 'skeleton':
        net = HedNet(n_channels=3, n_classes=1, bilinear=False, n_features=32)
        if args.loss == 'mae':
            checkpoint = "checkpoints/model_ensemble_snet.pth"
        elif args.loss == 'mse':
            checkpoint = "checkpoints/model_snet_inner_mse.pth"
        elif args.loss == 'smooth':
            checkpoint = "checkpoints/model_snet_inner_smooth.pth"
        
        
    device = get_device()
    net.to(device=device)
    #net.load_state_dict(torch.load(checkpoint))
    
    trainer = Trainer(net, device, test_ofda_subset=(args.testing_subset == "ofda"))
    trainer.net.load_state_dict(torch.load(checkpoint))

    #mae, mse = trainer.test(batch_size=5, printlog=False)
    inputs, gt, output = trainer.test_output(batch_size=5)

    print(inputs.shape)
    print(type(inputs))
    
    #imagen = make_grid(output, nrow=1, padding=0, normalize=True)
    #imagen = make_grid(gt, nrow=1, padding=0, normalize=True)
    #plt.imshow(imagen.permute(1,2,0))
    
    plt.figure(figsize=(1*5,3*5))
    
    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=0, hspace=0.1)
    
    
    imagen = make_grid(inputs, nrow=1, padding=0, normalize=True)
    plt.subplot(gs[0]), plt.axis('off'), plt.title('(a)', y=-0.05), plt.imshow(imagen.permute(1,2,0))
    
    imagen = make_grid(gt, nrow=1, padding=0, normalize=True)
    plt.subplot(gs[1]), plt.axis('off'), plt.title('(b)', y=-0.05), plt.imshow(imagen.permute(1,2,0))
    
    imagen = make_grid(output, nrow=1, padding=0, normalize=True)
    plt.subplot(gs[2]), plt.axis('off'), plt.title('(c)', y=-0.05), plt.imshow(imagen.permute(1,2,0))
    
    
    plt.savefig("tmp-output.png", bbox_inches='tight', pad_inches=0)
#    plt.show()


    
