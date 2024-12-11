import torch
from torch.utils.data import DataLoader

from dataset import cifar_datasets
from model import CNN

model_path = 'cifar_cnn.pth'

_, test_data = cifar_datasets()
test_loader = DataLoader(test_data, batch_size = 64, shuffle = False)

model = CNN()

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

if __name__ == "__main__":
    val_acc = 0

    with torch.no_grad():
        for images, labels in test_loader:
            test_outputs = model(images)
            val_acc += (test_outputs.max(1)[1] == labels).sum().item()
    avg_val_acc = val_acc / len(test_loader.dataset)

    print('Accuracy: {:.4f}'.format(avg_val_acc))