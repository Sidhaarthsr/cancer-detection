import torch
from torch import nn, optim

def train_loop(model, train_loader, criterion, optimizer, device):
    train_loss = 0.0
    total = 0
    correct = 0
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        train_loss += loss.item()
    train_acc = correct / total

    return train_acc*100, train_loss/len(train_loader)

def validation(model, val_loader, criterion, device):
    total = 0
    correct = 0
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss += loss.item()
    val_acc = correct / total

    return val_acc*100, val_loss/len(val_loader)