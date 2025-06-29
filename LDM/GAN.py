# --- Imports ---
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns
import colorsys
import json
import torch
sns.set_style("ticks")

import librosa
import librosa.display
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
import torchvision

print(f"Cuda available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# --- Main Data Loading ---
# Bare in mind that the data loading and paths may need to be adjusted based on your local setup.
def get_data(data_dir = "MusicBench/", base_name = "MusicBench", suffixes = ["train", "test_B"]):
    data = {}
    for suffix in suffixes:
        json_file = data_dir + base_name + "_" + suffix + ".json"
        print(json_file)
        data[suffix] = []
        with open(json_file, 'r') as f:
            for line in f:
                data[suffix].append(json.loads(line))
    return data

all_data = get_data()
all_data["test"] = all_data.pop("test_B")
audio_folder = "MusicBench/datashare"

# Dataset
class AutoencoderDataset(Dataset):
    def __init__(self, data_json, audio_folder, sr=16000, n_mels=128, fmax=8000, cut_secs=30):
        self.data = data_json
        self.audio_folder = audio_folder
        self.sr = sr
        self.n_mels = n_mels
        self.fmax = fmax
        self.cut_len = sr * cut_secs
        self.GLOBAL_MIN_DB = -80.0 
        self.GLOBAL_MAX_DB = 0.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            audio_path = os.path.join(self.audio_folder, item["location"])
            mix, _ = librosa.load(audio_path, sr=self.sr)
        except Exception as e:
            # On error, we'll just grab the next sample
            return self.__getitem__((idx + 1) % len(self))

        if len(mix) > self.cut_len:
            start = (len(mix) - self.cut_len) // 2
            end = start + self.cut_len
            mix = mix[start:end]

        mix_mel = librosa.feature.melspectrogram(y=mix, sr=self.sr, n_mels=self.n_mels, fmax=self.fmax)
        mix_db = librosa.power_to_db(mix_mel, ref=np.max)


        # Global Normalization ---
        denominator = self.GLOBAL_MAX_DB - self.GLOBAL_MIN_DB
    
        # Small number to avoid division by zero
        epsilon = 1e-6 
        mix_db_normalized = (mix_db - self.GLOBAL_MIN_DB) / (denominator + epsilon)
        mix_db_normalized = np.clip(mix_db_normalized, 0, 1)

        return torch.tensor(mix_db_normalized, dtype=torch.float32)


def collate_spectrograms(batch):
    max_len = max([x.shape[1] for x in batch])
    # Pad with 0 as our data is normalized between 0 and 1
    padded_spectrograms = [
        torch.nn.functional.pad(x, (0, max_len - x.shape[1]), "constant", 0.0)
        for x in batch
    ]
    return torch.stack(padded_spectrograms)

# This is a function to check for NaN values in the output of each layer. It helps to debug issues during training.
def nan_checker_hook(module, input, output):
    if torch.isnan(output).any():
        print(f"!!! NaN detected in the output of layer: {module} !!!")

# --- Model Definition ---

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.silu = nn.SiLU()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.silu(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.silu(h)
        h = self.conv2(h)
        
        return self.nin_shortcut(x) + h

class Autoencoder(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        # --- Encoder  ---
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            ResnetBlock(64, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Downsample
            ResnetBlock(128, 128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # Downsample
            ResnetBlock(256, 256),
            nn.Conv2d(256, latent_channels, kernel_size=3, stride=2, padding=1) # Final compression
        )

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, kernel_size=3, stride=1, padding=1),
            ResnetBlock(256, 256),
            # Upsample Block 1
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            ResnetBlock(128, 128),
            # Upsample Block 2
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            ResnetBlock(64, 64),
            # Upsample Block 3
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        original_size = x.shape[2:] 
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return F.interpolate(decoded, size=original_size, mode='bilinear', align_corners=False)
    
# --- Discriminator ---
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Wrap convolutional layers with spectral_norm for training stability
        self.model = nn.Sequential(
            # Layer 1
            torch.nn.utils.spectral_norm(
                nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            torch.nn.utils.spectral_norm(
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            torch.nn.utils.spectral_norm(
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4
            torch.nn.utils.spectral_norm(
                nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Output Layer
            torch.nn.utils.spectral_norm(
                nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            )
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1) 
        return self.model(x)
    
# --- Training ---

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Hyperparameters ---
config = {
    "EPOCHS": 50,
    "LEARNING_RATE": 1e-3,
    "G_LEARNING_RATE": 2e-4,
    "D_LEARNING_RATE": 1e-5,
    "BATCH_SIZE": 32,
    "LATENT_CHANNELS": 4,
    "CUT_SECS": 30,
    "N_MELS": 128,
    #"FMAX": 8000,
    "FMAX": 10000, 
    "DATA_SUBSET_PERCENT": 1.0 # Use 10% of the full dataset for training tests, it's way too big otherwise (+2 days)
}

# --- Create a unique directory for this training run ---
run_name = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join("GAN_runs", run_name)
os.makedirs(run_dir, exist_ok=True)
print(f"All artifacts for this run will be saved in: {run_dir}")


# --- Setup for Training ---

# Create subset of the dataset
train_data_full = all_data['train']
num_train_samples = int(len(train_data_full) * config["DATA_SUBSET_PERCENT"])
train_subset = train_data_full[:num_train_samples] # Slice the list
print(f"Using {num_train_samples} samples for training ({config['DATA_SUBSET_PERCENT']*100:.0f}% of the full training set).")

test_data_full = all_data['test']
num_test_samples = int(len(test_data_full) * config["DATA_SUBSET_PERCENT"])
test_subset = test_data_full[:num_test_samples] # Slice the list
print(f"Using {num_test_samples} samples for validation.")

# DataLoaders
ae_train_dataset = AutoencoderDataset(train_subset, audio_folder, cut_secs=config["CUT_SECS"])
train_loader = DataLoader(ae_train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, collate_fn=collate_spectrograms, num_workers=4, pin_memory=True)
ae_test_dataset = AutoencoderDataset(test_subset, audio_folder, cut_secs=config["CUT_SECS"])
test_loader = DataLoader(ae_test_dataset, batch_size=config["BATCH_SIZE"], collate_fn=collate_spectrograms, num_workers=4, pin_memory=True)

# Model, Loss, and Optimizer
autoencoder = Autoencoder(latent_channels=config["LATENT_CHANNELS"]).to(device)
discriminator = Discriminator().to(device)

optimizer_G = torch.optim.AdamW(autoencoder.parameters(), lr=config["G_LEARNING_RATE"], betas=(0.5, 0.999))
optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=config["D_LEARNING_RATE"], betas=(0.5, 0.999))

scheduler_G = ReduceLROnPlateau(optimizer_G, 'min', factor=0.5, patience=5)
scheduler_D = ReduceLROnPlateau(optimizer_D, 'min', factor=0.5, patience=5)

# 3. Define loss functions
l1_loss_fn = nn.L1Loss().to(device)
adversarial_loss_fn = nn.BCEWithLogitsLoss().to(device)

# Lambda for weighting the L1 loss vs the adversarial loss
lambda_l1 = 100.0

# Scaler for mixed precision
scaler_G = GradScaler('cuda')
scaler_D = GradScaler('cuda')

# --- Save the configuration and model architecture to a file ---
config_path = os.path.join(run_dir, "config_and_architecture.txt")
with open(config_path, 'w') as f:
    f.write("--- Training Configuration ---\n")
    json.dump(config, f, indent=4)
    f.write("\n\n--- Generator Architecture ---\n")
    f.write(str(autoencoder))
    f.write("\n\n--- Discriminator Architecture ---\n")
    f.write(str(discriminator))
print(f"Configuration and architecture saved to {config_path}")

EPOCHS = config["EPOCHS"]

history = {
    "G_loss": [],
    "D_loss": [],
    "recon_loss": [], # L1 loss component of G_loss
    "adv_loss": [],   # Adversarial component of G_loss
    "val_recon_loss": []
}

for name, module in autoencoder.named_modules():
    module.register_forward_hook(nan_checker_hook)

print("NaN checker hooks registered on the autoencoder.")

# --- Training Loop ---
print(f"Starting adversarial training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    autoencoder.train()
    discriminator.train()
    
    total_G_loss = 0
    total_D_loss = 0
    total_recon_loss = 0
    total_adv_loss = 0

    # --- Training Phase ---
    for spectrogram_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        spectrogram_batch = spectrogram_batch.to(device, non_blocking=True)
        real_spectrograms = spectrogram_batch.unsqueeze(1)

        b, _, h, w = real_spectrograms.shape
        dummy_output_shape = discriminator(torch.randn(1, 1, h, w, device=device)).shape
        
        # --- Smoothed labels for stability ---
        real_labels = (torch.ones(b, *dummy_output_shape[1:]) * 0.9).to(device) # Use 0.9 for real
        fake_labels = (torch.zeros(b, *dummy_output_shape[1:]) + 0.1).to(device) # Use 0.1 for fake

        # -----------------------
        #  Train Discriminator
        # -----------------------
        optimizer_D.zero_grad(set_to_none=True)
        
        with torch.no_grad(): # Use no_grad for the generator pass to save computation
             fake_spectrograms = autoencoder(spectrogram_batch).detach()

        with autocast('cuda'):
            loss_D_real = adversarial_loss_fn(discriminator(real_spectrograms), real_labels)
            loss_D_fake = adversarial_loss_fn(discriminator(fake_spectrograms), fake_labels)
            loss_D = (loss_D_real + loss_D_fake) / 2
            
        scaler_D.scale(loss_D).backward()
        # --- Clip discriminator gradients ---
        scaler_D.unscale_(optimizer_D)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)

        scaler_D.step(optimizer_D)
        scaler_D.update()
        
        # -----------------------
        #  Train Generator
        # -----------------------
        optimizer_G.zero_grad(set_to_none=True)

        with autocast('cuda'):
            reconstructed_spectrograms = autoencoder(spectrogram_batch)
            pred_gen = discriminator(reconstructed_spectrograms)
            
            loss_G_adv = adversarial_loss_fn(pred_gen, real_labels)
            loss_G_l1 = l1_loss_fn(reconstructed_spectrograms, real_spectrograms)
            loss_G = loss_G_adv + (lambda_l1 * loss_G_l1)

        scaler_G.scale(loss_G).backward()

        # --- Clip generator gradients ---
        scaler_G.unscale_(optimizer_G)
        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)

        scaler_G.step(optimizer_G)
        scaler_G.update()
        
        # --- Accumulate losses ---
        total_G_loss += loss_G.item()
        total_D_loss += loss_D.item()
        total_recon_loss += loss_G_l1.item() * lambda_l1 
        total_adv_loss += loss_G_adv.item()

    # --- Store average epoch losses ---
    history["G_loss"].append(total_G_loss / len(train_loader))
    history["D_loss"].append(total_D_loss / len(train_loader))
    history["recon_loss"].append(total_recon_loss / len(train_loader))
    history["adv_loss"].append(total_adv_loss / len(train_loader))

    # -----------------------
    #  Validation Phase
    # -----------------------
    autoencoder.eval() 
    total_val_recon_loss = 0
    
    with torch.no_grad(): 
        for i, val_batch in enumerate(tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validate]")):
            val_batch = val_batch.to(device, non_blocking=True)
            real_val_spectrograms = val_batch.unsqueeze(1)
            
            with autocast('cuda'):
                reconstructed_val = autoencoder(val_batch)
                val_recon_loss = l1_loss_fn(reconstructed_val, real_val_spectrograms)
            
            total_val_recon_loss += val_recon_loss.item()
            
            # --- Save one batch of reconstructed images for visual inspection ---
            if i == 0:
                comparison_grid = torch.cat([real_val_spectrograms[:8], reconstructed_val[:8]])
                val_image_path = os.path.join(run_dir, f"reconstruction_epoch_{epoch+1:03d}.png")
                torchvision.utils.save_image(comparison_grid, val_image_path, nrow=8, normalize=True)

    avg_val_loss = total_val_recon_loss / len(test_loader)
    history["val_recon_loss"].append(avg_val_loss)
    # --- Step the schedulers ---
    scheduler_G.step(avg_val_loss)
    scheduler_D.step(avg_val_loss)

    # --- Print epoch summary ---
    print(f"\nEPOCH [{epoch+1}/{EPOCHS}] "
          f"G_loss: {history['G_loss'][-1]:.4f} | "
          f"D_loss: {history['D_loss'][-1]:.4f} | "
          f"Val_Recon_Loss: {history['val_recon_loss'][-1]:.4f}")
    

print("\nAdversarial training complete. Saving artifacts...")

# 1. Save model weights (Generator and Discriminator)
torch.save(autoencoder.state_dict(), os.path.join(run_dir, "autoencoder.pth"))
torch.save(discriminator.state_dict(), os.path.join(run_dir, "discriminator.pth"))
print("Model weights saved.")

# 2. Save loss history
history_path = os.path.join(run_dir, "training_history.json")
with open(history_path, 'w') as f:
    json.dump(history, f)
print("Loss history saved.")

# 3. Save plot of losses
plt.figure(figsize=(12, 8))
plt.plot(history['G_loss'], label='Generator Total Loss')
plt.plot(history['D_loss'], label='Discriminator Loss')
plt.plot(history['recon_loss'], label='Reconstruction (L1) Loss', linestyle='--')
plt.plot(history['adv_loss'], label='Adversarial Loss', linestyle='--')
plt.plot(history['val_recon_loss'], label='Validation Reconstruction Loss', linewidth=3, marker='o')
plt.title(f"Training & Validation Losses - Run {run_name}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(run_dir, "loss_plot.png"), dpi=300)
plt.close()
print("Loss plot saved.")