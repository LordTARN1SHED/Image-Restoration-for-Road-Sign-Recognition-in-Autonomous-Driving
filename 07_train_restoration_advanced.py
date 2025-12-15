# 07_train_restoration_advanced.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import os
from tqdm import tqdm
from pathlib import Path

# ================= Configuration Area =================
# Specifically to address the drop in Blur scores, we only run Blur this time
TASK_NAME = 'Blur'  

# Parameter Settings
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0002 # Lower the learning rate slightly because the Loss has become more complex
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Weight Settings (Weight for Perceptual Loss)
LAMBDA_PERCEPTUAL = 0.1  # 0.1 is a classic empirical value

# Path Configuration
CLEAN_DIR = Path('./data/gtsrb/GTSRB/Training')
DISTORTED_DIR = Path(f'./data/processed/{TASK_NAME}')
SAVE_MODEL_PATH = f'./restoration_{TASK_NAME.lower()}.pth'
# ===================================================

print(f"=== Advanced Training Mode (Perceptual Loss) ===")
print(f"Current Task: {TASK_NAME}")
print(f"Goal: Force generation of sharp edges to improve VGG recognition rate")

# 1. Define Paired Dataset (Keep unchanged)
class PairedDataset(Dataset):
    def __init__(self, clean_root, distorted_root, transform=None):
        self.clean_root = clean_root
        self.distorted_root = distorted_root
        self.transform = transform
        self.clean_files = sorted(list(clean_root.glob('*/*.ppm')))
        self.data_pairs = []
        for c_path in self.clean_files:
            rel_path = c_path.relative_to(clean_root)
            d_path = distorted_root / rel_path
            if not d_path.exists():
                d_path = d_path.with_suffix('.png')
            if d_path.exists():
                self.data_pairs.append((d_path, c_path))
        print(f"Successfully matched image pairs: {len(self.data_pairs)}")

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

# 2. Define Restoration Network (Keep unchanged)
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.final = nn.Conv2d(64, 3, 1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        d2 = self.up2(b)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return out

# ================= New: Define Perceptual Loss =================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        # Load VGG16 feature extraction part
        vgg = models.vgg16(weights='DEFAULT').features
        # We only take the first 16 layers (contains the first few conv blocks, enough to extract textures and edge features)
        self.slice = nn.Sequential()
        for x in range(16):
            self.slice.add_module(str(x), vgg[x])
        
        # Freeze parameters, do not participate in training
        self.slice.eval()
        for param in self.slice.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # Calculate the distance between the generated image and the original image in VGG feature space
        return torch.mean((self.slice(x) - self.slice(y)) ** 2)
# ===============================================================

def train_model():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = PairedDataset(CLEAN_DIR, DISTORTED_DIR, transform=transform)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = SimpleUNet().to(DEVICE)
    
    # === Core Modification: Define two Losses ===
    criterion_pixel = nn.L1Loss() # Use L1 instead of MSE; L1 produces sharper images
    criterion_perceptual = VGGPerceptualLoss().to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for bad_imgs, clean_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            bad_imgs, clean_imgs = bad_imgs.to(DEVICE), clean_imgs.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(bad_imgs)
            
            # === Calculate Combined Loss ===
            loss_pixel = criterion_pixel(outputs, clean_imgs)
            loss_perceptual = criterion_perceptual(outputs, clean_imgs)
            
            # Total Loss = Pixel Difference + 0.1 * VGG Feature Difference
            loss = loss_pixel + LAMBDA_PERCEPTUAL * loss_perceptual
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.6f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bad_imgs, clean_imgs in val_loader:
                bad_imgs, clean_imgs = bad_imgs.to(DEVICE), clean_imgs.to(DEVICE)
                outputs = model(bad_imgs)
                # Can also look at total Loss during validation
                l_pix = criterion_pixel(outputs, clean_imgs)
                l_perc = criterion_perceptual(outputs, clean_imgs)
                val_loss += (l_pix + LAMBDA_PERCEPTUAL * l_perc).item()
                
        print(f"Epoch {epoch+1} Val Loss: {val_loss/len(val_loader):.6f}")
        
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), SAVE_MODEL_PATH)

    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"Training finished! Advanced model saved as {SAVE_MODEL_PATH}")

if __name__ == '__main__':
    train_model()