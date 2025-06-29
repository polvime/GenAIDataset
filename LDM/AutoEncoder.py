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

import librosa
import librosa.display
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
sns.set_style("ticks")

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
    def __init__(self, data_json, audio_folder, sr=22050, n_mels=128, fmax=8000, cut_secs=30):
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

        # Global Normalization
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


# --- Autoencoder Model Definition ---
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
        # --- Encoder ---
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
    
# --- Setup for Training ---

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Hyperparameters ---
config = {
    "EPOCHS": 20,
    "LEARNING_RATE": 1e-3,
    "BATCH_SIZE": 32,
    "LATENT_CHANNELS": 4,
    "CUT_SECS": 30,
    "N_MELS": 128,
    "FMAX": 8000,
    "DATA_SUBSET_PERCENT": 0.1 # Use 10% of the full dataset for training tests, it's way too big otherwise
}

# --- Create a unique directory for this training run ---
run_name = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join("training_runs", run_name)
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
l1_loss_fn = nn.L1Loss().to(device)
mse_loss_fn = nn.MSELoss().to(device)
optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=config["LEARNING_RATE"], weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
scaler = GradScaler('cuda') # For mixed precision

# --- Save the configuration and model architecture to a file ---
config_path = os.path.join(run_dir, "config_and_architecture.txt")
with open(config_path, 'w') as f:
    f.write("--- Training Configuration ---\n")
    json.dump(config, f, indent=4)
    f.write("\n\n--- Model Architecture ---\n")
    f.write(str(autoencoder))
print(f"Configuration and architecture saved to {config_path}")


# --- Training Loop ---
train_loss_history = []
test_loss_history = []
EPOCHS = config["EPOCHS"]

for epoch in range(EPOCHS):
    autoencoder.train()
    total_train_loss = 0
    for spectrogram_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        if torch.isnan(spectrogram_batch).any() or torch.isinf(spectrogram_batch).any():
            print("Skipping a batch because it contains invalid values (NaN or Inf)!")
            continue
        spectrogram_batch = spectrogram_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        with autocast('cuda'):
            reconstructed_batch = autoencoder(spectrogram_batch)
            target_batch = spectrogram_batch.unsqueeze(1)
            #loss = l1_loss_fn(reconstructed_batch, target_batch) + mse_loss_fn(reconstructed_batch, target_batch)
            loss = l1_loss_fn(reconstructed_batch, target_batch) * 0.7 + mse_loss_fn(reconstructed_batch, target_batch) * 0.3
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_train_loss += loss.item()
        
    avg_train_loss = total_train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    # --- Validation Phase ---
    autoencoder.eval()
    total_test_loss = 0
    with torch.no_grad():
        for spectrogram_batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Test]"):
            if torch.isnan(spectrogram_batch).any() or torch.isinf(spectrogram_batch).any():
                print("Skipping a batch because it contains invalid values (NaN or Inf)!")
                continue
            spectrogram_batch = spectrogram_batch.to(device, non_blocking=True)
            
            with autocast('cuda'):
                reconstructed_batch = autoencoder(spectrogram_batch)
                target_batch = spectrogram_batch.unsqueeze(1)
                loss = l1_loss_fn(reconstructed_batch, target_batch) + mse_loss_fn(reconstructed_batch, target_batch)
            
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_loss_history.append(avg_test_loss)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
    scheduler.step(avg_test_loss)

# --- Save the trained model ---
if not os.path.exists("Weights"):
    os.makedirs("Weights")

if os.path.exists("Weights/autoencoder_weights.pth"):
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    torch.save(autoencoder.state_dict(), f"Weights/autoencoder_weights_{timestamp}.pth")
else:
    torch.save(autoencoder.state_dict(), 'Weights/autoencoder_weights.pth')

# --- Save all artifacts ---
print("\nAutoencoder training complete. Saving artifacts...")

# 1. Save model weights
weights_path = os.path.join(run_dir, "autoencoder_weights.pth")
torch.save(autoencoder.state_dict(), weights_path)
print(f"Model weights saved to {weights_path}")

# 2. Save loss history
train_loss_path = os.path.join(run_dir, "train_loss_history.json")
test_loss_path = os.path.join(run_dir, "test_loss_history.json")
with open(train_loss_path, 'w') as f:
    json.dump(train_loss_history, f)
with open(test_loss_path, 'w') as f:
    json.dump(test_loss_history, f)
print(f"Loss histories saved to {run_dir}")

# 3. Save plot
plot_path = os.path.join(run_dir, "loss_plot.png")
plt.figure(figsize=(10, 6))
plt.plot(range(1, config["EPOCHS"] + 1), train_loss_history, marker='o', label='Training Loss')
plt.plot(range(1, config["EPOCHS"] + 1), test_loss_history, marker='x', label='Test Loss')
plt.title(f"Training & Test Loss - Run {run_name}")
plt.xlabel("Epoch")
plt.ylabel("Average MSE Loss")
plt.grid(True)
tick_spacing = max(1, config["EPOCHS"] // 10)
plt.xticks(np.arange(1, config["EPOCHS"] + 1, tick_spacing))
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Loss plot saved to {plot_path}")

