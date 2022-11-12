import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, BatchNorm2d, ReLU, BatchNorm1d


class GeoNet(nn.Module):

    def __init__(self):
        super(GeoNet, self).__init__()
        self.conv1 = Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = Conv2d(128, 64, kernel_size=3, padding=1)
        self.model = nn.Sequential(
            self.conv1,
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            self.conv2,
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            self.conv3,
            BatchNorm2d(128),
            ReLU(),
            self.conv4,
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            Linear(64 * 32 * 32, 256),
            BatchNorm1d(256),
            ReLU(),
            Linear(256, 8)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':  ##习惯在这个地方测试模型的正确性
    model = GeoNet()
    print(model)
    input = torch.ones((16, 1, 256, 256))
    output = model(input)
    print(output.shape)
