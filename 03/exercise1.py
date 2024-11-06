import torch
import numpy as np

data = np.array([
    [[65, 70], [56, 80], [78, 68], [90, 85], [60, 75]],
    [[70, 75], [54, 88], [82, 64], [88, 83], [58, 78]],
    [[67, 72], [52, 82], [80, 66], [86, 80], [59, 74]]])

print("===== problem 1 =====")
tensor1 = torch.tensor(data, dtype = float)
print(tensor1.size())
print("===== problem 2 =====")
tensor2 = torch.permute(tensor1, (2, 0, 1))
print(tensor2)
print(tensor2.size())
print("===== problem 3 =====")
tensor3 = torch.sum(tensor2, 0)
print(tensor3)
print("===== problem 4 =====")
print(torch.mean(tensor3, 1))
print("===== problem 5 =====")
print(torch.sum(tensor3, 1) / tensor3.size(1))