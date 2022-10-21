import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image

def normalize1(data):
    t_min = np.min(data)
    t_max = np.max(data)
    #t_min = 10.0
    #t_max = 50.0
    data[data < t_min] = t_min
    data[data > t_max] = t_max
    image = np.uint8((data - t_min) * 255 / (t_max-t_min))
    image.shape = (24,32)
    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return image

def normalize2(data):
    data = data * 10 - 100
    data[data < 0] = 0
    data[data > 255] = 255
    image = np.uint8(data)
    image.shape = (24,32)
    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return image

def normalize3(data):
    norm = np.linalg.norm(data)
    data = data / norm
    image = np.uint8(data * 255)
    image.shape = (24,32)
    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return image

def normalize4(data):
    # convert to grayscale image
    #data[data < 25] = 0
    #data[data > 35] = 0
    data = data * 10 - 100
    data[data < 0] = 0
    data[data > 255] = 255
    img = Image.new('L', (32, 24))
    img.putdata(data)
    return img

class ThermalImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, classes=None):
        labels = pd.read_csv(annotations_file)
        if isinstance(classes, int):
            labels = labels[labels[labels.columns[1]] == classes]
        self.img_labels = labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_idx = self.img_labels.iloc[idx, 0]
        data_file = os.path.join(self.img_dir, 'THERM_{:05d}.RAW'.format(img_idx))
        img_data = np.load(data_file)

        image = normalize4(img_data)

        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def index2name(self, idx):
        return '{:05d}'.format(self.img_labels.iloc[idx, 0])
