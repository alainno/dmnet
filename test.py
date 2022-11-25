import argparse
import torch
import torch.nn as nn
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from unet import UNet

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--architecture', '-a', type=str, default="UNet", help='Train with UNet')
    return parser.parse_args()

# def net(input):
#     return torch.randn(2, 1, 256, 256)

def get_test_loader(batch_size=2):
    test_img_path = "/Users/alain/Documents/desarrollo/dmnet/datasets/synthetic/test/images/"
    test_gt_path = "/Users/alain/Documents/desarrollo/dmnet/datasets/synthetic/test/masks/"

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


def test(model, batch_size=2, printlog=False):
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
        #target = net(input)
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, ground_truth)
        test_loss += loss.item()

    test_loss /= len(test_loader)
    return test_loss


if __name__=='__main__':
    args = get_args()

    #if args.architecture == 'SkeletonNet':
    # guardar la curva de pérdida

    print('Testing with U-Net...')

    model = UNet(n_channels=3,n_classes=1,bilinear=False,n_features=64)
    
    model.load_state_dict(torch.load('checkpoints/model.pth'))

    test_loss = test(model, batch_size=4, printlog=True)

    print('Test L1 Loss:', test_loss)