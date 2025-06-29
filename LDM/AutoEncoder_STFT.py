print("starting AutoEncoderResNet_STFT.py...")
# --- Imports ---
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import librosa
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

# --- Setup ---
print(f"Cuda available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# --- STFT Loss Implementation ---

# Class to calculate the loss for a single STFT resolution
class STFTLoss(nn.Module):
    def __init__(self, fft_size=1024, hop_size=256, win_size=1024):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.window = torch.hann_window(self.win_size).to(device)
        self.l1_loss = nn.L1Loss()

    def forward(self, y_hat, y):
        # y_hat and y are raw audio waveforms
        stft_hat = torch.stft(y_hat.squeeze(1), n_fft=self.fft_size, hop_length=self.hop_size, win_length=self.win_size, window=self.window, return_complex=True)
        stft_true = torch.stft(y.squeeze(1), n_fft=self.fft_size, hop_length=self.hop_size, win_length=self.win_size, window=self.window, return_complex=True)
        
        # Calculate spectral convergence and log magnitude losses
        sc_loss = torch.norm(torch.abs(stft_true) - torch.abs(stft_hat), p="fro") / torch.norm(torch.abs(stft_true), p="fro")
        mag_loss = self.l1_loss(torch.log(torch.abs(stft_true) + 1e-7), torch.log(torch.abs(stft_hat) + 1e-7))
        
        return sc_loss + mag_loss

# Class to combine multiple STFT losses at different resolutions
class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512], win_sizes=[512, 1024, 2048]):
        super().__init__()
        self.loss_layers = nn.ModuleList([STFTLoss(f, h, w) for f, h, w in zip(fft_sizes, hop_sizes, win_sizes)])

    def forward(self, y_hat, y):
        total_loss = 0
        for layer in self.loss_layers:
            total_loss += layer(y_hat, y)
        return total_loss / len(self.loss_layers)
    
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

class AutoencoderDataset(Dataset):
    def __init__(self, data_json, audio_folder, sr=22050, n_mels=128, fmax=8000, cut_secs=30):
        self.data = data_json
        self.audio_folder = audio_folder
        self.sr = sr
        self.n_mels = n_mels
        self.fmax = fmax
        self.cut_len = sr * cut_secs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            audio_path = os.path.join(self.audio_folder, item["location"])
            mix, _ = librosa.load(audio_path, sr=self.sr)

            # Pad or truncate audio
            if len(mix) > self.cut_len:
                start = (len(mix) - self.cut_len) // 2
                mix = mix[start:start+self.cut_len]
            else:
                mix = np.pad(mix, (0, self.cut_len - len(mix)), 'constant')

            if np.max(np.abs(mix)) == 0: return None
            
            mix_mel = librosa.feature.melspectrogram(y=mix, sr=self.sr, n_mels=self.n_mels, fmax=self.fmax)
            mix_db = librosa.power_to_db(mix_mel, ref=np.max)
            
            # Get min and max for this specific spectrogram
            min_db, max_db = mix_db.min(), mix_db.max()

            if (max_db - min_db) == 0: return None
            
            normalized_db = (mix_db - min_db) / (max_db - min_db)
            
            # Return min/max values along with the data
            return {
                "waveform": torch.tensor(mix, dtype=torch.float32),
                "spectrogram": torch.tensor(normalized_db, dtype=torch.float32),
                "min_db": min_db,
                "max_db": max_db
            }
        
        except Exception:
            return None

def collate_fn_stft(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    
    waveforms = torch.stack([item['waveform'] for item in batch])
    spectrograms = torch.stack([item['spectrogram'] for item in batch])
    # Also batch the min/max values
    min_dbs = torch.tensor([item['min_db'] for item in batch], dtype=torch.float32)
    max_dbs = torch.tensor([item['max_db'] for item in batch], dtype=torch.float32)
    
    return {
        "spectrograms": spectrograms,
        "waveforms": waveforms,
        "min_dbs": min_dbs,
        "max_dbs": max_dbs
    }


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
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            ResnetBlock(64, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Downsample
            ResnetBlock(128, 128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # Downsample
            ResnetBlock(256, 256),
            nn.Conv2d(256, latent_channels, kernel_size=3, stride=2, padding=1) # Final compression
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, kernel_size=3, stride=1, padding=1),
            ResnetBlock(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), # Upsample
            ResnetBlock(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), # Upsample
            ResnetBlock(64, 64),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2), # Upsample
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return F.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=False)
    
# --- Setup for Training ---

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Hyperparameters ---
config = {
    "EPOCHS": 16,
    "LEARNING_RATE": 1e-3,
    "BATCH_SIZE": 4,
    "LATENT_CHANNELS": 4,
    "CUT_SECS": 30,
    "N_MELS": 128,
    "FMAX": 8000,
    "DATA_SUBSET_PERCENT": 0.1 
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
train_loader = DataLoader(ae_train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn_stft, num_workers=4, pin_memory=True)
ae_test_dataset = AutoencoderDataset(test_subset, audio_folder, cut_secs=config["CUT_SECS"])
test_loader = DataLoader(ae_test_dataset, batch_size=config["BATCH_SIZE"], collate_fn=collate_fn_stft, num_workers=4, pin_memory=True)

# Model and Optimizer
autoencoder = Autoencoder(latent_channels=config["LATENT_CHANNELS"]).to(device)
optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=config["LEARNING_RATE"], weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
scaler = GradScaler('cuda')  # For mixed precision training

# Loss functions and inverse transform
l1_loss_fn = nn.L1Loss().to(device)
stft_loss_fn = MultiResolutionSTFTLoss().to(device)
griffin_lim = T.GriffinLim(n_fft=2048, hop_length=512).to(device)

# --- Save the configuration and model architecture to a file ---
config_path = os.path.join(run_dir, "config_and_architecture.txt")
with open(config_path, 'w') as f:
    f.write("--- Training Configuration ---\n")
    json.dump(config, f, indent=4)
    f.write("\n\n--- Model Architecture ---\n")
    f.write(str(autoencoder))
print(f"Configuration and architecture saved to {config_path}")

inverse_mel_transform = T.InverseMelScale(n_stft=1025, n_mels=128, sample_rate=22050).to(device)

# --- Training Loop ---
train_loss_history = []
test_loss_history = []
EPOCHS = config["EPOCHS"] 

for epoch in range(EPOCHS):
    autoencoder.train()
    total_train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        if batch is None: continue
        
        specs_true = batch["spectrograms"].to(device, non_blocking=True)
        waveforms_true = batch["waveforms"].to(device, non_blocking=True)
        min_dbs = batch["min_dbs"].to(device, non_blocking=True)
        max_dbs = batch["max_dbs"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda'):
            reconstructed_spec_norm = autoencoder(specs_true.unsqueeze(1)).squeeze(1)
            
            # --- Denormalization ---
            min_vals = min_dbs.unsqueeze(1).unsqueeze(1)
            max_vals = max_dbs.unsqueeze(1).unsqueeze(1)
            reconstructed_spec_db = reconstructed_spec_norm * (max_vals - min_vals) + min_vals
            
            # --- Convert to Waveform ---
            reconstructed_power = torch.pow(10.0, reconstructed_spec_db / 10.0)
            reconstructed_linear = inverse_mel_transform(reconstructed_power)
            reconstructed_waveform = griffin_lim(reconstructed_linear) 
            
            # --- Ensure waveform lengths match before STFT loss ---
            n = min(waveforms_true.size(1), reconstructed_waveform.size(1))
            waveforms_true_trimmed = waveforms_true[:, :n]
            reconstructed_waveform_trimmed = reconstructed_waveform[:, :n]
            
            # --- Calculate Combined Loss on trimmed waveforms ---
            loss_l1 = l1_loss_fn(reconstructed_spec_norm, specs_true)
            loss_stft = stft_loss_fn(reconstructed_waveform_trimmed.unsqueeze(1), waveforms_true_trimmed.unsqueeze(1))
            
            loss = loss_l1 + (2.5 * loss_stft)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_train_loss += loss.item()
        
    avg_train_loss = total_train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    # --- Validation Phase ---
    autoencoder.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Test]"):
            if batch is None: continue

            specs_true = batch["spectrograms"].to(device, non_blocking=True)
            waveforms_true = batch["waveforms"].to(device, non_blocking=True)
            min_dbs = batch["min_dbs"].to(device, non_blocking=True)
            max_dbs = batch["max_dbs"].to(device, non_blocking=True)
            
            with autocast(device_type='cuda'):
                reconstructed_spec_norm = autoencoder(specs_true.unsqueeze(1)).squeeze(1)
                
                # Denormalization
                min_vals = min_dbs.unsqueeze(1).unsqueeze(1)
                max_vals = max_dbs.unsqueeze(1).unsqueeze(1)
                reconstructed_spec_db = reconstructed_spec_norm * (max_vals - min_vals) + min_vals
                
                # Convert to Waveform
                reconstructed_power = torch.pow(10.0, reconstructed_spec_db / 10.0)
                reconstructed_linear = inverse_mel_transform(reconstructed_power)
                reconstructed_waveform = griffin_lim(reconstructed_linear)

                # --- Trim waveforms ---
                n = min(waveforms_true.size(1), reconstructed_waveform.size(1))
                waveforms_true_trimmed = waveforms_true[:, :n]
                reconstructed_waveform_trimmed = reconstructed_waveform[:, :n]

                # Calculate Combined Loss
                loss_l1 = l1_loss_fn(reconstructed_spec_norm, specs_true)
                loss_stft = stft_loss_fn(reconstructed_waveform_trimmed.unsqueeze(1), waveforms_true_trimmed.unsqueeze(1))
                loss = loss_l1 + (2.5 * loss_stft)

            total_test_loss += loss.item()

    if len(test_loader) > 0:
        avg_test_loss = total_test_loss / len(test_loader)
        test_loss_history.append(avg_test_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
        scheduler.step(avg_test_loss)
    else:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Test Loss: N/A")

# --- Save All Artifacts to the Run Directory ---
print(f"\nAutoencoder training complete. Saving artifacts to {run_dir}...")

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
plt.figure(figsize=(12, 7))
plt.plot(range(1, EPOCHS + 1), train_loss_history, marker='o', label='Training Loss')
plt.plot(range(1, EPOCHS + 1), test_loss_history, marker='x', label='Test Loss')
plt.title(f"Training & Test Loss - Run {run_name}", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Combined Loss (L1 + STFT)", fontsize=12) # Updated label
plt.grid(True)
tick_spacing = max(1, EPOCHS // 10)
plt.xticks(np.arange(1, EPOCHS + 1, tick_spacing))
plt.legend()
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Loss plot saved to {plot_path}")