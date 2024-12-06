import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import cifar_datasets
from model import CNN

train_data, test_data = cifar_datasets()
train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 64, shuffle = False)

model = CNN()

criterion = nn.CrossEntropyLoss()

learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

epochs = 2

if __name__ == "__main__":
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()

            train_outputs = model(images)
            loss = criterion(train_outputs, labels)
            train_loss += loss.item()
            train_acc += (train_outputs.max(1)[1] == labels).sum().item()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                test_outputs = model(images)
                loss = criterion(test_outputs, labels)
                print(labels)
                val_loss += loss.item()
                val_acc += (test_outputs.max(1)[1] == labels).sum().item()
            avg_val_loss = val_loss / len(test_loader)
            avg_val_acc = val_acc / len(test_loader.dataset)

        print("Epoch {}, Loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch + 1, avg_train_loss, avg_val_loss, avg_val_acc))