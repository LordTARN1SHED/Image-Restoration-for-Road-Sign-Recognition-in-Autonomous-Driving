# 07_train_restoration.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from pathlib import Path

# ================= Configuration Area (Modify here for each run) =================
# Options: 'Noise', 'Blur', 'Fog'
# Please modify here and run three times respectively!
TASK_NAME = 'Fog'  

# Parameter Settings
BATCH_SIZE = 32
EPOCHS = 15            # 15 epochs are enough to see obvious results
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path Configuration
CLEAN_DIR = Path('./data/gtsrb/GTSRB/Training')
DISTORTED_DIR = Path(f'./data/processed/{TASK_NAME}')
SAVE_MODEL_PATH = f'./restoration_{TASK_NAME.lower()}.pth'
# ============================================================================

print(f"Current Task: Training [{TASK_NAME}] restoration model")
print(f"Input Data: {DISTORTED_DIR}")
print(f"Target Data: {CLEAN_DIR}")
print(f"Model Save Path: {SAVE_MODEL_PATH}")

# 1. Define Paired Dataset (Input distorted image, label is clean image)
class PairedDataset(Dataset):
    def __init__(self, clean_root, distorted_root, transform=None):
        self.clean_root = clean_root
        self.distorted_root = distorted_root
        self.transform = transform
        
        # Find all image files
        self.clean_files = sorted(list(clean_root.glob('*/*.ppm')))
        # Ensure files exist in corresponding folder (only take matching filenames)
        self.data_pairs = []
        for c_path in self.clean_files:
            # Construct corresponding distorted image path
            rel_path = c_path.relative_to(clean_root)
            d_path = distorted_root / rel_path
            
            # Your generation script saves as png, some might be ppm, compatibility check here
            if not d_path.exists():
                d_path = d_path.with_suffix('.png') # Try finding png
            
            if d_path.exists():
                self.data_pairs.append((d_path, c_path))
        
        print(f"Successfully matched image pairs: {len(self.data_pairs)} images")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        d_path, c_path = self.data_pairs[idx]
        
        bad_img = Image.open(d_path).convert('RGB')
        clean_img = Image.open(c_path).convert('RGB')
        
        if self.transform:
            bad_img = self.transform(bad_img)
            clean_img = self.transform(clean_img)
            
        return bad_img, clean_img

# 2. Define Restoration Network (Simplified U-Net)
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        
        # Encoder (Downsampling)
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        
        # Decoder (Upsampling)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        
        # Output layer
        self.final = nn.Conv2d(64, 3, 1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder (with Skip Connections)
        d2 = self.up2(b)
        d2 = torch.cat((d2, e2), dim=1) # Skip Connection
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1) # Skip Connection
        d1 = self.dec1(d1)
        
        out = self.final(d1)
        return out

def train_model():
    # Data Preprocessing (Input/Target must be same size)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load Data
    dataset = PairedDataset(CLEAN_DIR, DISTORTED_DIR, transform=transform)
    
    # Split Train/Val (90% train, 10% val)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize Model
    model = SimpleUNet().to(DEVICE)
    criterion = nn.MSELoss() # Pixel-level Loss, make output as close to original as possible
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for bad_imgs, clean_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            bad_imgs, clean_imgs = bad_imgs.to(DEVICE), clean_imgs.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(bad_imgs)
            loss = criterion(outputs, clean_imgs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss (MSE): {avg_loss:.6f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bad_imgs, clean_imgs in val_loader:
                bad_imgs, clean_imgs = bad_imgs.to(DEVICE), clean_imgs.to(DEVICE)
                outputs = model(bad_imgs)
                loss = criterion(outputs, clean_imgs)
                val_loss += loss.item()
        print(f"Epoch {epoch+1} Val Loss: {val_loss/len(val_loader):.6f}")
        
        # Save checkpoint every 5 epochs, will also save at the end
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), SAVE_MODEL_PATH)

    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"Training finished! Model saved as {SAVE_MODEL_PATH}")

if __name__ == '__main__':
    train_model()