""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, n_features=64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, n_features) #190x198
        self.down1 = Down(n_features, n_features*2) #95x99
        self.down2 = Down(n_features*2, n_features*2*2) # 48x50
        self.down3 = Down(n_features*2*2, n_features*2*2*2)# 24x25
        factor = 2 if bilinear else 1
        self.down4 = Down(n_features*2*2*2, (n_features*2*2*2*2) // factor) #12x13
        self.up1 = Up(n_features*2*2*2*2, (n_features*2*2*2) // factor, bilinear) #24x26
        self.up2 = Up(n_features*2*2*2, (n_features*2*2) // factor, bilinear) #48x52
        self.up3 = Up(n_features*2*2, (n_features*2) // factor, bilinear) #96x104
        self.up4 = Up(n_features*2, n_features, bilinear) #192x208
        self.outc = OutConv(n_features, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
