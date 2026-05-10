import torch
from torch import nn


class Residual(nn.Module):
    """A ResNet basic block.

    Two 3x3 convolutions with batch norm and ReLU, plus a skip connection.
    When the spatial size or channel count of the input does not match the
    output of the second conv, the shortcut path uses a 1x1 conv (selected
    by `use_1conv=True`) to match shapes before the additive residual.
    """

    def __init__(self, input_channels, num_channels, use_1conv=False, stride=1):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
        )
        self.conv2 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=num_channels,
                kernel_size=1,
                stride=stride,
            )
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        return self.relu(y + x)


class ResNet18(nn.Module):
    """ResNet-18 (He et al., 2015), adapted to single-channel input.

    The structure is a 7x7 stem, four stages of two BasicBlocks at channel
    widths [64, 128, 256, 512], global average pool, and a single linear
    head. The first block in stages 2, 3, 4 downsamples with stride=2 and
    uses a 1x1 shortcut conv to match channels. Stage 1 keeps stride=1
    because the stem already downsamples twice.
    """

    def __init__(self, in_channels=1, num_classes=10):
        super(ResNet18, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = nn.Sequential(
            Residual(64, 64, use_1conv=False, stride=1),
            Residual(64, 64, use_1conv=False, stride=1),
        )
        self.stage2 = nn.Sequential(
            Residual(64, 128, use_1conv=True, stride=2),
            Residual(128, 128, use_1conv=False, stride=1),
        )
        self.stage3 = nn.Sequential(
            Residual(128, 256, use_1conv=True, stride=2),
            Residual(256, 256, use_1conv=False, stride=1),
        )
        self.stage4 = nn.Sequential(
            Residual(256, 512, use_1conv=True, stride=2),
            Residual(512, 512, use_1conv=False, stride=1),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    print(summary(model, input_size=(1, 224, 224)))
