import argparse
import torch
import torch.nn as nn
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from unet import UNet
from hednet import HedNet

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet and SkeletonNet on images and target masks')
    parser.add_argument('-a', '--architecture', type=str, choices=["unet","skeleton"], default="unet", help='Architecture UNet or SkeletonNet')
    return parser.parse_args()

# def net(input):
#     return torch.randn(2, 1, 256, 256)

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

    model.eval()

    test_loss = 0
    for batch in test_loader:
        input, ground_truth = batch['image'],batch['mask']
        input = input.to(device=device, dtype=torch.float32)
        ground_truth = ground_truth.to(device=device, dtype=torch.float32)

        #target = net(input)
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, ground_truth)
        test_loss += loss.item()

    test_loss /= len(test_loader)
    return test_loss

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device


if __name__=='__main__':
    args = get_args()

    if args.architecture == 'unet':
        print("Iniciando el test con U-Net...")
        net = UNet(n_channels=3, n_classes=1, bilinear=False, n_features=64)
    elif args.architecture == 'skeleton':
        print("Test con Skeleton-Net...")
        net = HedNet(n_channels=3, n_classes=1, bilinear=False, side=4, n_features=64)
   
    device = get_device()
    net.to(device=device)
    net.load_state_dict(torch.load('checkpoints/model_unet_2022.pth'))

    test_loss = test(net, device, batch_size=4, printlog=True)

    print('Test L1 Loss:', test_loss)