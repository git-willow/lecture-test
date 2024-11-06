import torch
from torch.nn import Module

class MyModel(Module):
    def __init__(self, mytensor, elem_add, elem_multiply):
        super().__init__()
        self.mytensor = mytensor
        self.elem_add = elem_add
        self.elem_multiply = elem_multiply

    def forward(self, x):
        out1 = x + self.mytensor
        out2 = out1 + self.elem_add
        out3 = out2 * self.elem_multiply
        return out1, out2, out3

if __name__ == "__main__":
    mymodel = MyModel(torch.ones((3, 3)), 4, 6)
    p2out, p3out, p4out = mymodel(torch.full((3, 3), 2))
    print("===== problem 2 =====")
    print(repr(p2out))
    print("===== problem 3 =====")
    print(repr(p3out))
    print("===== problem 4 =====")
    print(repr(p4out))