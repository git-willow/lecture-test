import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import cifar_datasets
from model import CNN

model_path = 'cifar_cnn.pth'

train_data, _ = cifar_datasets()
train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

if __name__ == "__main__":
    epochs = 20

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0

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

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss
        }, model_path)

    print('Epoch: {}, Loss: {loss:.4f}'.format(epoch + 1, loss = avg_train_loss))