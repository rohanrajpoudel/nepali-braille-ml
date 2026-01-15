import os, cv2, torch, json
import numpy as np
from torch.utils.data import Dataset
from utils import generate_gaussian_map, preprocess_image, parse_annotation

class BrailleDataset(Dataset):
    def __init__(self, list_path, base_dir, sigma=2, img_size=(512, 512), augment=False):
        with open(list_path) as f:
            self.samples = [line.strip() for line in f if line.strip()]
        self.base_dir = base_dir
        self.sigma = sigma
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path = self.samples[idx]
        img_path = os.path.join(self.base_dir, rel_path + "+recto.jpg")
        ann_path = os.path.join(self.base_dir, rel_path + "+recto.txt")

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(img_path)

        dots = parse_annotation(ann_path, img.shape)
        gt = generate_gaussian_map(img.shape, dots, sigma=self.sigma)
        img = preprocess_image(img, self.img_size)
        gt = cv2.resize(gt, self.img_size, interpolation=cv2.INTER_LINEAR)

        img = torch.from_numpy(img).unsqueeze(0).float()
        gt = torch.from_numpy(gt).unsqueeze(0).float()
        return img, gt
