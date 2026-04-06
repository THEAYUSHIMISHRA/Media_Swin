import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils.degradation import degrade_image

class XRayDataset(Dataset):

    def __init__(self, root_dir):

        self.images = []

        covid_path = os.path.join(root_dir,"COVID")
        normal_path = os.path.join(root_dir,"Normal")

        for img in os.listdir(covid_path):
            self.images.append(os.path.join(covid_path,img))

        for img in os.listdir(normal_path):
            self.images.append(os.path.join(normal_path,img))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Scales [0, 1] to [-1, 1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        path = self.images[idx]

        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

        clean = img.copy()
        degraded = degrade_image(img)

        clean = self.transform(clean)
        degraded = self.transform(degraded)

        clean = clean.repeat(3,1,1)
        degraded = degraded.repeat(3,1,1)

        return {
            "clean": clean,
            "degraded": degraded
        }