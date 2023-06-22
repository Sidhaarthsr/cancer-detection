import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms

data_dir = "./LungColon"

class ImageDataset(Dataset):
    def __init__(self, data_paths, labels, transform=None):
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        image_path = self.data_paths[index]
        label = self.labels[index]
        image = cv2.imread(image_path)
        #image = cv2.resize(image, image_size)
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data(data_dir = data_dir, batch_size = 128,num_workers = 2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224,224))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])

    data_paths = []
    labels = []

    # Define the desired image size
    image_size = (256, 256)

    # Iterate through each folder in the dataset directory
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            # Iterate through each image file in the folder
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                if os.path.isfile(image_path):
                    #print(image_path)
                    # Add the image path to the data_paths list
                    data_paths.append(image_path)
                    # Add the label (parent folder name) to the labels list
                    labels.append(folder_name)

    # Perform label encoding on the class labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Split the dataset into training and testing sets
    train_data_paths, test_data_paths, train_labels, test_labels = train_test_split(
        data_paths, labels, test_size=0.2, random_state=42)

    # Create PyTorch datasets and data loaders
    train_dataset = ImageDataset(train_data_paths, train_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=num_workers, shuffle=True)

    val_dataset = ImageDataset(test_data_paths, test_labels, transform=val_transform)
    val_loader = DataLoader(val_dataset,num_workers=num_workers, batch_size=batch_size)

    return train_loader, val_loader
    
