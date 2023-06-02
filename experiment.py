import os
import torch
from torch import nn, optim
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from data_loader import load_data
from model import Modelv1, Modelv2
import timeit
from util import train_loop, validation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flag = torch.cuda.device_count()
model = Modelv1()
if flag>1:
    model = nn.DataParallel(model)
model.to(device)
lr = [.01, .001, .0001]
bs = [32, 64, 128, 256, 512]
print("Starting Training")
for x in bs:
    train_loader, val_loader = load_data(data_dir = "./LungColon", batch_size = x, num_workers = 8)
    for y in lr:
        print(f"Using Learning Rate : {y},  Using Batch Size : {x}")
        criterion = nn.CrossEntropyLoss()
        if flag>1:
            optimizer = optim.Adam(model.module.parameters(), lr=y)
        else:
            optimizer = optim.Adam(model.parameters(), lr=y)

        num_epochs = 50
        best_val_acc = 0
        best_model_path = "./model"
        for epoch in range(num_epochs):
            start = timeit.default_timer()
            train_acc, train_loss = train_loop(model, train_loader, criterion, optimizer, device)
            val_acc, val_loss = validation(model, val_loader, criterion, device)
            end = timeit.default_timer()

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Time : {end-start}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if flag>1:
                    torch.save(model.module.state_dict(), f'model_{x}_{y}.pth')
                else:
                    torch.save(model.state_dict(), f'model_{x}_{y}.pth')

        print("Training complete!")
        print(f"Best Val Accuracy {best_val_acc}")