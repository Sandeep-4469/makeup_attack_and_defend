import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from models.generator import UNetGenerator
from models.discriminator import PatchDiscriminator
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset ---
class MakeupDataset(Dataset):
    def __init__(self, root_makeup, root_clean, transform=None):
        self.makeup_images = sorted(os.listdir(root_makeup))
        self.clean_images = sorted(os.listdir(root_clean))
        self.root_makeup = root_makeup
        self.root_clean = root_clean
        self.transform = transform

    def __len__(self):
        return min(len(self.makeup_images), len(self.clean_images))

    def __getitem__(self, idx):
        makeup = Image.open(os.path.join(self.root_makeup, self.makeup_images[idx])).convert("RGB")
        clean = Image.open(os.path.join(self.root_clean, self.clean_images[idx])).convert("RGB")

        if self.transform:
            makeup = self.transform(makeup)
            clean = self.transform(clean)
        return makeup, clean

# --- Hyperparameters ---
img_size = 256
batch_size = 8
lr = 2e-4
num_epochs = 100
lambda_l1 = 100  # for L1 loss weighting

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# --- Load Dataset ---
dataset = MakeupDataset("data/makeup", "data/non-makeup", transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Models ---
generator = UNetGenerator().to(device)
discriminator = PatchDiscriminator().to(device)

# --- Losses ---
criterion_GAN = nn.MSELoss()
criterion_L1 = nn.L1Loss()

# --- Optimizers ---
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# --- Training Loop ---
for epoch in range(num_epochs):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for i, (real_A, real_B) in enumerate(pbar):
        real_A = real_A.to(device)  # With makeup
        real_B = real_B.to(device)  # Without makeup

        valid = torch.ones((real_A.size(0), 1, 30, 30), requires_grad=False).to(device)
        fake = torch.zeros((real_A.size(0), 1, 30, 30), requires_grad=False).to(device)

        # --- Train Generator ---
        optimizer_G.zero_grad()
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)
        loss_L1 = criterion_L1(fake_B, real_B)
        loss_G = loss_GAN + lambda_l1 * loss_L1
        loss_G.backward()
        optimizer_G.step()

        # --- Train Discriminator ---
        optimizer_D.zero_grad()
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        pbar.set_postfix(G=loss_G.item(), D=loss_D.item())

    # --- Save Images ---
    if (epoch + 1) % 5 == 0:
        os.makedirs("outputs", exist_ok=True)
        fake_B = (fake_B + 1) / 2  # De-normalize
        torchvision.utils.save_image(fake_B[:4], f"outputs/fake_{epoch+1}.png", nrow=2)

    # --- Save Model Checkpoints ---
    if (epoch + 1) % 10 == 0:
        torch.save(generator.state_dict(), f"checkpoints/generator_epoch{epoch+1}.pth")
        torch.save(discriminator.state_dict(), f"checkpoints/discriminator_epoch{epoch+1}.pth")
