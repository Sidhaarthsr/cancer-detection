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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Specify the path to your image dataset
dataset_path = "./LungColon"

# Define a custom dataset class
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
        image = cv2.resize(image, image_size)
        if self.transform:
            image = self.transform(image)
        return image, label


# Initialize empty lists for storing image data paths and labels
data_paths = []
labels = []

# Define the desired image size
image_size = (256, 256)

# Iterate through each folder in the dataset directory
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    if os.path.isdir(folder_path):
        # Iterate through each image file in the folder
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if os.path.isfile(image_path):
                print(image_path)
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
train_dataset = ImageDataset(train_data_paths, train_labels, transform=ToTensor())
test_dataset = ImageDataset(test_data_paths, test_labels, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the model architecture
num_classes = 5
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 64 * 64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Create an instance of the model
model = Net()

# Move the model and data to GPU if available
model.to(device)

# Print if computation is running on CPU or GPU
print("Computation Device:", device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print the average training loss for each epoch
    print(f"Epoch {epoch+1} - Loss: {running_loss / len(train_loader)}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate the accuracy on the test set
accuracy = correct / total
print("Accuracy:", accuracy)
