import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split

seeding = 42

transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.CenterCrop(192),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class FROGdata(Dataset):
    def __init__(self, csv_file, img_dir_shg,img_dir_thg, transforms = transforms, train=True, val_size=0.2):
        self.data = pd.read_csv(csv_file, header=None)
        self.img_dir_shg = img_dir_shg
        self.img_dir_thg = img_dir_thg
        self.transforms = transforms

        train_indices, val_indices = train_test_split(
            range(len(self.data)),
            test_size=val_size,
            random_state=seeding,
        )

        if train:
            self.data = self.data.iloc[train_indices]
        else:
            self.data = self.data.iloc[val_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_filename_shg = self.data.iloc[index,0]
        img_filename_thg = img_filename_shg.replace('shg', 'thg')

        img_path_shg = os.path.join(self.img_dir_shg, img_filename_shg)
        img_path_thg = os.path.join(self.img_dir_thg, img_filename_thg)

        try:
            image_shg = Image.open(img_path_shg).convert('RGB')
            image_thg = Image.open(img_path_thg).convert('RGB')

        except Exception as e:
            print(f"Error opening images: {img_path_shg} or {img_path_thg}. Exception: {e}")
            return None, None

        if self.transforms:
            image_shg = self.transforms(image_shg)
            image_thg = self.transforms(image_thg)

        dual_channel_image = torch.cat((image_shg, image_thg), dim=0)
        points = torch.tensor(self.data.iloc[index, 64:192].tolist(), dtype=torch.float32)

        return dual_channel_image, points, img_filename_shg

class single_data(Dataset):
    def __init__(self, data_path, img_dir, transforms=transforms, train = True):
        self.data = pd.read_csv(data_path, header=None)
        self.img_dir = img_dir
        self.transforms = transforms

        train_indices, val_indices = train_test_split(
            range(len(self.data)),
            test_size=0.2,
            random_state=seeding,
        )

        if train:
            self.data = self.data.iloc[train_indices]

        else:
            self.data = self.data.iloc[val_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        index = self.data.iloc[idx, 0]
        img_filename = f"{index}"
        img_path = os.path.join(self.img_dir, img_filename)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image: {img_path}. Exception: {e}")
            return None, None

        if self.transforms:
            image = self.transforms(image)

        points = torch.tensor(self.data.iloc[idx, 64:192].tolist(), dtype=torch.float32)

        return image, points, img_filename
