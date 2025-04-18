""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from .hednet_parts import *
from .coordconv import CoordConv1d, CoordConv2d, CoordConv3d, AddCoords
import numpy as np

class HedNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, side=-1, n_features=64, use_cuda=0):
        super(HedNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.side = side

        #self.coordconv = CoordConv2d(n_channels, n_features, 1, use_cuda=use_cuda)
        self.coordconv = AddCoords(2, False, use_cuda=use_cuda)
        #self.inc = DoubleConv(n_features, n_features)
        self.inc = DoubleConv(n_channels + 2, n_features)
        self.down1 = Down2(n_features, n_features*2, use_cuda=use_cuda)
        self.down2 = Down2(n_features*2, n_features*4, use_cuda=use_cuda)
        self.down3 = Down2(n_features*4, n_features*8, use_cuda=use_cuda)
        self.down4 = Down2(n_features*8, n_features*16, use_cuda=use_cuda)
        self.inc2 = DoubleConv(n_features*16, n_features*32)
        self.up1 = Up(n_features*32, n_features*8, False)
        self.up2 = Up(n_features*8, n_features*4, False)
        self.up3 = Up(n_features*4, n_features*2, False)
        self.up4 = Up(n_features*2, n_features, False)
        
        #self.upso1 = nn.ConvTranspose2d(n_features*8, 1, kernel_size=8, stride=8)
        #self.upso2 = nn.ConvTranspose2d(n_features*4, 1, kernel_size=4, stride=4)
        #self.upso3 = nn.ConvTranspose2d(n_features*2, 1, kernel_size=2, stride=2)
        #self.upso4 = nn.ConvTranspose2d(n_features, 1, kernel_size=1, stride=1)
        
        self.upso1 = UpSide(n_features*8, 1, kernel_size=8, stride=8)
        self.upso2 = UpSide(n_features*4, 1, kernel_size=4, stride=4)
        self.upso3 = UpSide(n_features*2, 1, kernel_size=2, stride=2)
        self.upso4 = UpSide(n_features, 1, kernel_size=1, stride=1)
        
        self.dilation = nn.Conv2d(4 + 2, 1, kernel_size=3, padding=2, dilation=2)
        

    def forward(self, x):
        x = self.coordconv(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.inc2(x)
        so1 = self.up1(x, x4)
        so2 = self.up2(so1, x3)
        so3 = self.up3(so2, x2)
        so4 = self.up4(so3, x1)
        
        #upsample1 = self.upso1(so1)
        #upsample2 = self.upso2(so2)
        #upsample3 = self.upso3(so3)
        #upsample4 = self.upso4(so4)

        upsample1 = self.upso1(so1, x1)
        upsample2 = self.upso2(so2, x1)
        upsample3 = self.upso3(so3, x1)
        upsample4 = self.upso4(so4, x1)
        
        fuse = torch.cat((upsample1, upsample2, upsample3, upsample4), dim=1)
        fuse = self.coordconv(fuse) # nuevo
        fuse = self.dilation(fuse)
        
        ensembled = upsample4 * 0.5 # upsample4 is side 1
        ensembled = ensembled.add(fuse * 0.5)
        
        results = [upsample1, upsample2, upsample3, upsample4, fuse, ensembled]
        #results = [upsample1, upsample2, upsample3, upsample4, ensembled]
        
        return results[self.side]
        
        #return x
        #return fuse
    
        '''
        x0 = self.inc(x)
        x1 = self.down0(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x, x0)
        logits = self.outc(x)        
        return logits
        '''
        
class EnsembleSkeletonNet(nn.Module):
    def __init__(self, model1, model2):
        super(EnsembleSkeletonNet, self).__init__()
        self.model1 = model1
        self.model2 = model2
        #self.avg = nn.AvgPool2d(2)
        
    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        #out = self.avg(torch.cat([out1,out2], dim=1))
        out = out1 * 0.5
        out = out.add(out2 * 0.5)
        return out