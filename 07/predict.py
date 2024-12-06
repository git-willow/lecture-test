import torch
from torch.utils.data import DataLoader

from dataset import cifar_datasets
from model import CNN

train_data, test_data = cifar_datasets()
train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 64, shuffle = False)

model = CNN()

train_outputs_list = []
test_outputs_list = []

for epoch in range(1):
    model.train()
    for images, labels in train_loader:
        train_outputs = model(images)
        train_outputs_list.append(train_outputs)

    model.eval()
    with torch.no.grad():
        for images, labels in test_loader:
            test_outputs = model(images)
            test_outputs_list.append(test_outputs)

if __name__ == "__main__":
    print("Train Output Size: ", train_outputs_list[0].shape)
    print("Test Output Size: ", test_outputs_list[0].shape)