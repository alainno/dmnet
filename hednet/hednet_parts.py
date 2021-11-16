""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

from .se_module import SELayer
from .squeeze_and_excitation import ChannelSpatialSELayer, ChannelSELayer

class RSBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        #self.sb = SELayer(out_channels)
        self.sb = ChannelSELayer(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    
    def forward(self, x):
        residual = x
        out = self.relu(x)
        out = self.bn1(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sb(out)
        out += self.conv3(residual)
        return out
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, ks=3):
        super().__init__()
        '''
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=ks, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=ks, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        '''

        #self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        #self.convrelu1 = nn.Sequential(
        #    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        #    nn.ReLU(inplace=True)
        #)        
        self.rsblock1 = RSBlock(in_channels, out_channels)
        #self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        #self.convrelu2 = nn.Sequential(
        #    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        #    nn.ReLU(inplace=True)
        #)        
        self.rsblock2 = RSBlock(out_channels, out_channels)

    def forward(self, x):
        #return self.double_conv(x)
        #x = self.convrelu1(x)
        x = self.rsblock1(x)
        #x = self.convrelu2(x)
        x = self.rsblock2(x)
        return x
    
class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, ks=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=ks, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, None, kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=4)
            #self.conv = DoubleConv(in_channels, out_channels)
            #self.conv = SingleConv(in_channels // 2, out_channels)
            self.rsblock = RSBlock(out_channels * 2, out_channels)
            self.cs_se = ChannelSpatialSELayer(out_channels, out_channels)

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
        #return self.conv(x)
        x = self.rsblock(x)
        return self.cs_se(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1)

    def forward(self, x):
        return self.conv(x)

    
class UpSide(nn.Module):
    '''
    Upscaling the side
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        return x1