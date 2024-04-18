# Define a custom dataset class
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BasicDataset(Dataset):
    def __init__(self, data_dir: str, applied_transforms: transforms.Compose = None):
        self.data_dir = data_dir
        self.images = os.listdir(data_dir)
        self.transform = applied_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image
