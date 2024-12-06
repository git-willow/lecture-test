import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import cifar_datasets
from model import CNN

train_data, test_data = cifar_datasets()
train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 64, shuffle = False)

model = CNN()

criterion = nn.CrossEntropyLoss()

for epoch in range(1):
    train_loss = 0
    val_loss = 0

    model.train()
    for images, labels in train_loader:
        train_outputs = model(images)
        loss = criterion(train_outputs, labels)
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            test_outputs = model(images)
            loss = criterion(test_outputs, labels)
            val_loss += loss.item()
        avg_val_loss = val_loss / len(test_loader)

if __name__ == "__main__":
    print("Train loss: ", avg_train_loss)
    print("Validation loss: ", avg_val_loss)