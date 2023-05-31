import os
import torch
from torch import nn, optim
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.RandomCrop((224,224))
])

# Set the paths
data_dir = './LungColon'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# Load the datasets
train_dataset = ImageFolder(train_dir, transform=transform)
val_dataset = ImageFolder(val_dir, transform=transform)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#batch sizes {128, 256, 512}
val_loader = DataLoader(val_dataset, batch_size=128)

# Define the model architecture
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(256 * 28 * 28, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, len(train_dataset.classes))
)

# Define the loss function and optimizer
# Softmax is applied internally by nn.CrossEntropyLoss(), FocalLoss
criterion = nn.CrossEntropyLoss()
#learning_rate = 0.001, 0.1, 0.01
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
model.to(device)

num_epochs = 16
best_val_loss = float('inf')
best_model_path = "path/to/saved/model.pth"

for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0

    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

    train_loss /= len(train_dataset)
    val_loss /= len(val_dataset)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)

print("Training complete!")

# Load the best model
best_model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(256 * 28 * 28, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, len(train_dataset.classes))
)

best_model.load_state_dict(torch.load(best_model_path))
best_model.to(device)
