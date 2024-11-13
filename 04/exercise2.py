import torch
from torch import nn

print("===== problem 1 =====")
tensor1 = torch.ones((32, 1024))
print(tensor1.size())

print("===== problem 2 =====")
fc1 = nn.Linear(in_features = 1024, out_features = 256, bias = True)
out1 = fc1(tensor1)
print(out1.size())

print("===== problem 3 =====")
fc2 = nn.Linear(in_features = 1024, out_features = 2048, bias = True)
out2 = fc2(tensor1)
print(out2.size())

print("===== appendix =====")
print(out1.view((32, 16, 16)).size())