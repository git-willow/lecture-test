import torch
from models import MyModel

if __name__ == "__main__":
    tensor = torch.ones((32, 3, 128, 128))
    model = MyModel()
    out = model(tensor)
    print(out.size())
    