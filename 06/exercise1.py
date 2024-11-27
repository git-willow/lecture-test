from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

preprocess_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((250, 250)),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])

preprocess_2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.2)
])

preprocess_3 = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size = (250, 250), scale = (0.08, 1.0))
])

image_path = "./exercise_data/dog_img.png"
image = Image.open(image_path)
preprocessed_image = preprocess_1(image)
plt.imshow(preprocessed_image.permute(1, 2, 0))
plt.show()

preprocessed_image = preprocess_2(image)
plt.imshow(preprocessed_image.permute(1, 2, 0))
plt.show()

preprocessed_image = preprocess_3(image)
plt.imshow(preprocessed_image.permute(1, 2, 0))
plt.show()