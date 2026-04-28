import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(AlexNet, self).__init__(*args, **kwargs)
        self.relu = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)
        self.p2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.p4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.c6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.c7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.p8 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=6*6*256, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=10)

    def forward(self, x):
        x = self.relu(self.c1(x))
        x1 = self.p2(x)
        x2 = self.relu(self.c3(x1))
        x3 = self.p4(x2)
        x4 = self.relu(self.c5(x3))
        x5 = self.relu(self.c6(x4))
        x6 = self.relu(self.c7(x5))
        x7 = self.p8(x6)
        x8 = self.flatten(x7)
        x9 = self.relu(self.fc1(x8))
        x9 = F.dropout(x9, 0.5)
        x10 = self.relu(self.fc2(x9))
        x10 = F.dropout(x10, 0.5)
        x11 = self.fc3(x10)

        return x11
        

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlexNet().to(device)

    print(summary(model, input_size=(1, 227, 227)))

            
