import torch
from torch import nn
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sig = nn.Sigmoid()
        self.p2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.p4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc5 = nn.Linear(in_features=400, out_features=120)
        self.fc6 = nn.Linear(in_features=120, out_features=84)
        self.fc7 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x1 = self.sig(self.c1(x))
        x2 = self.p2(x1)
        x3 = self.sig(self.c3(x2))
        x4 = self.p4(x3)
        x5 = self.flatten(x4)
        x6 = self.sig(self.fc5(x5))
        x7 = self.sig(self.fc6(x6))
        x8 = self.fc7(x7)
        
        return x8
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LeNet().to(device)

    print(summary(model, input_size=(1, 28, 28)))