from model import SkelNet
from torchsummary import summary

net = SkelNet(1, 1, False)

#print(net)
summary(net, (1,256,256))