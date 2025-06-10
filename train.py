import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

from utils.dataset_utils import HAM10000Dataset
from models.cnn_model import CNN

def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return epoch_loss, accuracy

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss=0
    correct=0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()* images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss/len(loader.dataset)
    accuracy = correct/len(loader.dataset)
    return epoch_loss, accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")
    image_dir = "/Users/dhanalakshmijothi/Desktop/python/med_cnn_classifier/data/images"

    train_dataset = HAM10000Dataset(train_df, image_dir, transform)
    val_dataset = HAM10000Dataset(val_df, image_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch{epoch+1}/{num_epochs}")
        print(f"Train loss: {train_loss: 4f} | Train Acc: {train_acc:.4f}")
        print(f"Val loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth ")

if __name__=="__main__":
    main()
