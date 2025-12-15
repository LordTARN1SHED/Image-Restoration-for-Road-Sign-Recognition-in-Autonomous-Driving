# 06_test_baseline.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# ================= Configuration =================
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = './vgg16_baseline.pth'

# Test dataset path list
TEST_DIRS = {
    "Clean (Original)": './data/gtsrb/GTSRB/Training',
    "Noisy":            './data/processed/Noise',
    "Blurred":          './data/processed/Blur',
    "Foggy":            './data/processed/Fog'
}
# ===============================================

def evaluate_model(model, data_dir, name):
    """
    Evaluate model accuracy on the specified directory
    """
    if not os.path.exists(data_dir):
        print(f"Skipping {name}: Path does not exist {data_dir}")
        return

    # Must be the same preprocessing as training
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    # For speed or trend verification, we could take a subset, but here we run full amount
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    correct = 0
    total = 0
    
    model.eval()
    print(f"\nTesting: {name} ...")
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"Result [{name}] Accuracy: {acc*100:.2f}%")
    return acc

def main():
    print(f"Using device: {DEVICE}")

    # 1. Load model architecture
    model = models.vgg16(weights=None) # No need to download weights here since we load our own trained pth
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 43) # Must match the structure used during training
    
    # 2. Load weights
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found, please run 05_train_baseline.py first")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(DEVICE)
    print("Model loaded successfully!")

    # 3. Test sequentially
    results = {}
    for name, path in TEST_DIRS.items():
        acc = evaluate_model(model, path, name)
        if acc is not None:
            results[name] = acc

    # 4. Print final summary
    print("\n" + "="*30)
    print("Final Test Report (Baseline 1)")
    print("="*30)
    print(f"{'Dataset':<20} | {'Accuracy':<10}")
    print("-" * 32)
    for name, acc in results.items():
        print(f"{name:<20} | {acc*100:.2f}%")
    print("="*30)

if __name__ == '__main__':
    main()