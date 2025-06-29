# --- Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import json

# --- ResNet Autoencoder class ---
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
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            ResnetBlock(64, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            ResnetBlock(128, 128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            ResnetBlock(256, 256),
            nn.Conv2d(256, latent_channels, kernel_size=3, stride=2, padding=1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, kernel_size=3, stride=1, padding=1),
            ResnetBlock(256, 256),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            ResnetBlock(128, 128),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            ResnetBlock(64, 64),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        original_size = x.shape[2:]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return F.interpolate(decoded, size=original_size, mode='bilinear', align_corners=False)

# --- Data loading function ---
def get_data(data_dir = "MusicBench/", base_name = "MusicBench", suffixes = ["test_B"]):
    data = {}
    for suffix in suffixes:
        json_file = os.path.join(data_dir, f"{base_name}_{suffix}.json")
        print(f"Loading metadata from: {json_file}")
        data[suffix] = []
        with open(json_file, 'r') as f:
            for line in f:
                data[suffix].append(json.loads(line))
    return data

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000
N_MELS = 128
FMAX = 8000
CUT_SECS = 30
GLOBAL_MIN_DB = -80.0
GLOBAL_MAX_DB = 0.0
N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_INDEX = 50

# --- Define paths for BOTH models, this may need to be changed ---
AE_WEIGHTS_PATH = "Runs&Weights/AutoEncoder/autoencoder_weights.pth" # Standard Autoencoder
GAN_AE_WEIGHTS_PATH = "Runs&Weights/GAN/autoencoder.pth" # GAN-trained Autoencoder

# The dataset is not available on this repository, but you can download it.
audio_folder = "MusicBench/datashare"
all_data = get_data()
all_data["test"] = all_data.pop("test_B")


### Function to perform reconstruction
def reconstruct_spectrogram(weights_path, input_tensor, device):
    """Loads a model with given weights and reconstructs a spectrogram."""
    print(f"Loading weights from {weights_path}...")
    model = Autoencoder(latent_channels=4).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        reconstructed_normalized_tensor = model(input_tensor)
    
    reconstructed_normalized_db = reconstructed_normalized_tensor.squeeze().cpu().numpy()
    
    # Denormalize the output
    reconstructed_db = reconstructed_normalized_db * (GLOBAL_MAX_DB - GLOBAL_MIN_DB) + GLOBAL_MIN_DB
    return reconstructed_db

# --- Main Logic ---

# 1. Load and preprocess the original audio sample (only needs to be done once)
print("Loading and processing sample audio...")
test_item = all_data["test"][SAMPLE_INDEX]
audio_path = os.path.join(audio_folder, test_item['location'])
original_waveform, _ = librosa.load(audio_path, sr=SR)

# Cut and create spectrogram
cut_len = SR * CUT_SECS
if len(original_waveform) > cut_len:
    start = (len(original_waveform) - cut_len) // 2
    original_waveform = original_waveform[start:start + cut_len]

original_mel = librosa.feature.melspectrogram(y=original_waveform, sr=SR, n_mels=N_MELS, fmax=FMAX, n_fft=N_FFT, hop_length=HOP_LENGTH)
original_db = librosa.power_to_db(original_mel, ref=np.max)

# Normalize for model input
denominator = GLOBAL_MAX_DB - GLOBAL_MIN_DB
epsilon = 1e-6
mix_db_normalized = np.clip((original_db - GLOBAL_MIN_DB) / (denominator + epsilon), 0, 1)
input_tensor = torch.tensor(mix_db_normalized, dtype=torch.float32).unsqueeze(0).to(device)

# 2. Get reconstructions from both models
ae_reconstructed_db = reconstruct_spectrogram(AE_WEIGHTS_PATH, input_tensor, device)
gan_ae_reconstructed_db = reconstruct_spectrogram(GAN_AE_WEIGHTS_PATH, input_tensor, device)

### Create the comparative plot
print("Generating 1x3 comparative spectrogram plot...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True, layout='constrained')
fig.suptitle('Reconstruction Quality Comparison', fontsize=22)

# --- Plot 1: Original Spectrogram ---
librosa.display.specshow(original_db, sr=SR, x_axis='time', y_axis='mel', ax=axes[0], fmax=FMAX)
axes[0].set_title("Original", fontsize=16)
axes[0].set_ylabel('Frequency (Hz)', fontsize=14)
axes[0].set_xlabel('Time (s)', fontsize=14)

# --- Plot 2: Standard Autoencoder Reconstruction ---
librosa.display.specshow(ae_reconstructed_db, sr=SR, x_axis='time', y_axis='mel', ax=axes[1], fmax=FMAX)
axes[1].set_title("Autoencoder", fontsize=16)
axes[1].set_xlabel('Time (s)', fontsize=14)

# --- Plot 3: GAN-trained Autoencoder Reconstruction ---
img = librosa.display.specshow(gan_ae_reconstructed_db, sr=SR, x_axis='time', y_axis='mel', ax=axes[2], fmax=FMAX)
axes[2].set_title("GAN Autoencoder", fontsize=16)
axes[2].set_xlabel('Time (s)', fontsize=14)

# --- Colorbar ---
fig.colorbar(img, ax=axes.tolist(), format='%+2.0f dB', pad=0.01)

# --- Save and Show Plot ---
output_filename = "reconstruction_comparison.png"
plt.savefig(output_filename, dpi=300)
print(f"Plot saved to {output_filename}")
plt.show()