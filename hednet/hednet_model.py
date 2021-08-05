""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from .hednet_parts import *
from .coordconv import CoordConv1d, CoordConv2d, CoordConv3d
import numpy as np

class HedNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(HedNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.coordconv = CoordConv2d(n_channels, 32, 1)
        self.inc = DoubleConv(32, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.inc2 = DoubleConv(512, 1024)
        self.up1 = Up(1024, 256, False)
        self.up2 = Up(256, 128, False)
        self.up3 = Up(128, 64, False)
        self.up4 = Up(64, 32, False)
        
        self.upso1 = nn.ConvTranspose2d(256, 1, kernel_size=8, stride=8)
        self.upso2 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=4)
        self.upso3 = nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)
        self.upso4 = nn.ConvTranspose2d(32, 1, kernel_size=1, stride=1)
        
        self.dilation = nn.Conv2d(4, 1, kernel_size=3, padding=2, dilation=2)
        
        '''
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, 1) # bottleneck after down
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.up5 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        '''

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
        
    
        #weight_deconv4 =  make_bilinear_weights(16, 1).cuda()
        #weight_deconv2 =  make_bilinear_weights(4, 1).cuda()
        #upsample1 = torch.nn.functional.conv_transpose2d(so1, weight_deconv1, stride=2)
        #upsample5 = torch.nn.functional.conv_transpose2d(so5, weight_deconv5, stride=16)
        #upsample4 = torch.nn.functional.conv_transpose2d(so4, weight_deconv4, stride=8)
        
        upsample1 = self.upso1(so1)
        upsample2 = self.upso2(so2)
        upsample3 = self.upso3(so3)
        upsample4 = self.upso4(so4)
        
        fuse = torch.cat((upsample1, upsample2, upsample3, upsample4), dim=1)
        fuse = self.dilation(fuse)
        
        ensembled = upsample2 * 0.2
        ensembled = ensembled.add(fuse * 0.8)
        
        results = [upsample1, upsample2, upsample3, upsample4, fuse, ensembled]
        #results = [torch.sigmoid(r) for r in results]
        return results
        
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
        
