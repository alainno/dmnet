""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        padding = 0 if kernel_size == 1 else 1
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 1"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        padding = 0 if kernel_size == 1 else 1

        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)    

    
class QuadConv(nn.Module):
    """(convolution => [BN] => ReLU) * 4"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.quad_conv = nn.Sequential(
            SingleConv(in_channels, out_channels, kernel_size),
            SingleConv(out_channels, out_channels, kernel_size),
            SingleConv(out_channels, out_channels, kernel_size),
            SingleConv(out_channels, out_channels, kernel_size),
        )

    def forward(self, x):
        return self.quad_conv(x)
    
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            #DoubleConv(in_channels, out_channels, None, kernel_size)
            SingleConv(in_channels, out_channels, kernel_size),
            BasicBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            #self.conv = DoubleConv(in_channels, out_channels)
            self.conv = SingleConv(in_channels, out_channels)
            #self.conv = QuadConv(in_channels, out_channels)
            #self.resblock = BasicBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)
        #return self.resblock(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


# https://github.com/trailingend/pytorch-residual-block/blob/master/main.py
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()

        self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ) 
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(), # agregado
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity  = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += identity
        out = self.relu(out)
        return out

        #residual = self.conv_res(x)
        #x = self.conv_block1(x)
        #x = self.conv_block2(x)
        #x = x + residual
        #out = self.relu(x)
        #return out
