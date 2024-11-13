import torch
from torch import nn

print("===== problem 1 =====")
tensor1 = torch.ones((32, 3, 128, 128))
print(tensor1.size())

print("===== problem 2 =====")
conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3)
out1 = conv1(tensor1)
print(out1.size())

print("===== problem 3 =====")
conv2 = nn.Conv2d(in_channels = 3, out_channels = 256, kernel_size = 3, stride = 2, padding = 1)
out2 = conv2(tensor1)
print(out2.size())

print("===== problem 4-1 =====")
conv3 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 5, padding = 1)
out3 = conv3(tensor1)
print(out3.size())

print("===== problem 4-2 =====")
conv4 = nn.Conv2d(in_channels = 3, out_channels = 256, kernel_size = 5, stride = 2, padding = 2)
out4 = conv4(tensor1)
print(out4.size())