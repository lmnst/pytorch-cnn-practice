import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.ReLu = nn.ReLU()

        # 路线1. conv 1*1
        self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)

        # 路线2. conv1*1 3*3
        self.p2_1 = nn.Conv2d(in_channels=in_channels,out_channels=c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)

        # 路线3. conv1*1 5*5
        self.p3_1 = nn.Conv2d(in_channels=in_channels,out_channels=c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)

        # 路线4. Maxpool 3*3, Conv 1*1
        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1)
        self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)

    def forward(self, x):
        p1 = self.ReLu(self.p1_1(x))
        p2 = self.ReLu(self.p2_2(self.ReLu(self.p2_1(x))))
        p3 = self.ReLu(self.p3_2(self.ReLu(self.p3_1(x))))
        p4 = self.ReLu(self.p4_2(self.p4_1(x)))

        # 在通道的维度上进行连接，dim = 1
        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, Inception):
        super(GoogLeNet, self).__init__()
        self.b1 = nn.Sequential(
            
        )
