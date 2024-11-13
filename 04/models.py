import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = 3, out_channels = 256, kernel_size = 5, stride = 8)
        self.bn = nn.BatchNorm2d(num_features = 256, affine = False)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features = 256 * 16 * 16, out_features = 64, bias = True)

    def forward(self, x):
        # print("base:", x.size())
        x = self.conv(x)
        # print("conv:", x.size())
        x = self.bn(x)
        # print("bn:", x.size())
        x = self.relu(x)
        # print("relu:", x.size())
        x = x.view(-1, 256 * 16 * 16)
        # print("reshape:", x.size())
        x = self.fc(x)
        # print("linear:", x.size())
        return x