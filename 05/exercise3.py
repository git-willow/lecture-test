from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, dataset_dir):
        dir_path_resolved = Path(dataset_dir).resolve()
        dir_list = list(dir_path_resolved.glob("*"))
        dir_list = sorted(dir_list)
        self.img_list = []
        for dir in dir_list:
            self.img_list += list(dir.glob("*.png"))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path)
        img_tensor = self.transform(img)
        img_path = Path(img_path)
        parts = img_path.parts
        label = int(parts[-2])
        return img_tensor, label

if __name__ == "__main__":
    my_dataset = MyDataset("./data")
    img, label = my_dataset[0]
    print("===== problem1.1 =====")
    print(img.size())
    print("===== problem1.2 =====")
    print(label)