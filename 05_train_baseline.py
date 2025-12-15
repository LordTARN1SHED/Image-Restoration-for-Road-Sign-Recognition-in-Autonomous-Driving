# 05_train_baseline.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm

# ================= Configuration =================
BATCH_SIZE = 64
EPOCHS = 10           # 10 epochs are usually enough
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = './data/gtsrb/GTSRB/Training'
SAVE_PATH = './vgg16_baseline.pth'
# ===============================================

def train():
    print(f"Using device: {DEVICE}")

    # 1. Data Preprocessing
    # VGG16 default input is 224x224, GTSRB image sizes vary, must Resize
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet standard mean
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. Load Dataset
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=data_transforms)
    
    # Split 80% training, 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    print(f"Number of classes: {len(full_dataset.classes)}")

    # 3. Define Model (VGG16)
    # Use pretrained weights (weights='DEFAULT') to accelerate convergence
    model = models.vgg16(weights='DEFAULT')
    
    # Freeze earlier feature extraction layers (optional, here we fine-tune, so we don't freeze completely, but could lock first few layers)
    # For simplicity and utilizing your 4090 power, we use full parameter fine-tuning for best results
    
    # Modify the last fully connected layer (originally 1000 classes, changed to 43)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 43)
    
    model = model.to(DEVICE)

    # 4. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # 5. Training Loop
    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        # --- Training ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct / val_total
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Model saved, current best val acc: {best_acc:.4f}")

    print("Training complete!")

if __name__ == '__main__':
    train()