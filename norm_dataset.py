import os
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

seeding = 42

transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Normalize RGB not grayscale image to range [-1, 1]
])


class frogdata(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, train=True, val_size=0.2):
        self.data = pd.read_csv(csv_file, header=None)
        self.data += 1
        self.img_dir = img_dir
        self.transforms = transforms

        train_indices, val_indices = train_test_split(
            range(len(self.data)),
            test_size=val_size,
            random_state=seeding,
        )

        # if train:
        #     self.data = self.data.iloc[train_indices]
        # else:
        #     self.data = self.data.iloc[val_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # img_filename = self.data.iloc[index, 0]
        # img_path = os.path.join(self.img_dir, img_filename)
        # # print(f"Attempting to open image at path: {img_path}")  # Debug print
        #
        # try:
        #     image = Image.open(img_path).convert('RGB')
        # except Exception as e:
        #     print(f"Error opening image: {img_path}. Exception: {e}")
        #     return None, None
        img_path = os.path.join(self.img_dir, self.data.iloc[index, 0])
        image = Image.open(img_path).convert('RGB')
        points = torch.tensor(self.data.iloc[index, 175:335].tolist(), dtype=torch.float32)

        if self.transforms:
            image = self.transforms(image)

        return image, points


if __name__ == '__main__':
    pass
