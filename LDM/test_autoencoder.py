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
import soundfile as sf

# --- ResNet Autoencoder class from training ---
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
    
# --- Data loading function ---
# Bare in mind that the data loading and paths may need to be adjusted based on your local setup.
def get_data(data_dir = "MusicBench/", base_name = "MusicBench", suffixes = ["train", "test_B"]):
    data = {}
    for suffix in suffixes:
        json_file = data_dir + base_name + "_" + suffix + ".json"
        print(f"Loading metadata from: {json_file}")
        data[suffix] = []
        with open(json_file, 'r') as f:
            for line in f:
                data[suffix].append(json.loads(line))
    return data


# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#SR = 22050
SR = 16000 
N_MELS = 128
#FMAX = 10000
FMAX = 8000
CUT_SECS = 30
GLOBAL_MIN_DB = -80.0 
GLOBAL_MAX_DB = 0.0
N_FFT = 2048
HOP_LENGTH = 512

all_data = get_data()
all_data["test"] = all_data.pop("test_B")
audio_folder = "MusicBench/datashare"
WEIGHTS_PATH = "Runs&Weights/GAN/autoencoder.pth"
SAMPLE_INDEX = 50 #  Change this index to test different audio files


# --- Main ---
# 1. Load the trained model
print(f"Loading weights from {WEIGHTS_PATH}...")
model = Autoencoder(latent_channels=4).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()
print("Model loaded successfully.")

# 2. Get a sample from the test set
print("Loading test data...")
test_item = all_data["test"][SAMPLE_INDEX]

# 3. Preprocess the original audio exactly as during training
print(f"Loading and processing sample audio: {test_item['location']}")
audio_path = os.path.join(audio_folder, test_item['location'])
original_waveform, _ = librosa.load(audio_path, sr=SR)

# Cut from middle to match training
cut_len = SR * CUT_SECS
if len(original_waveform) > cut_len:
    start = (len(original_waveform) - cut_len) // 2
    end = start + cut_len
    original_waveform = original_waveform[start:end]

original_mel = librosa.feature.melspectrogram(
    y=original_waveform, 
    sr=SR, 
    n_mels=N_MELS, 
    fmax=FMAX,
    n_fft=N_FFT,          # <-- ADD THIS
    hop_length=HOP_LENGTH # <-- ADD THIS
)
original_db = librosa.power_to_db(original_mel, ref=np.max)

denominator = GLOBAL_MAX_DB - GLOBAL_MIN_DB

# Small number to avoid division by zero
epsilon = 1e-6
mix_db_normalized = (original_db - GLOBAL_MIN_DB) / (denominator + epsilon)
mix_db_normalized = np.clip(mix_db_normalized, 0, 1)

input_tensor =  torch.tensor(mix_db_normalized, dtype=torch.float32).unsqueeze(0).to(device)

# 4. Get the model's reconstruction
with torch.no_grad():
    reconstructed_normalized_tensor = model(input_tensor)
reconstructed_normalized_db = reconstructed_normalized_tensor.squeeze(0).squeeze(0).cpu().numpy()

# 5. Denormalize the output for visualization and audio conversion
reconstructed_db = reconstructed_normalized_db * (GLOBAL_MAX_DB - GLOBAL_MIN_DB) + GLOBAL_MIN_DB

# --- Part 1: Visual Evaluation ---
print("Generating spectrogram plot...")
fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, sharey=True, layout='constrained')
fig.suptitle('GAN Autoencoder Reconstruction Quality', fontsize=20)

# --- Plot 1: Original Spectrogram ---
librosa.display.specshow(original_db, sr=SR, x_axis='time', y_axis='mel', ax=axes[0], fmax=FMAX)
axes[0].set_title("Original Spectrogram", fontsize=16)
axes[0].set_ylabel('Frequency (Hz)', fontsize=16) 
axes[0].set_xlabel('')

# --- Plot 2: Reconstructed Spectrogram ---
img = librosa.display.specshow(reconstructed_db, sr=SR, x_axis='time', y_axis='mel', ax=axes[1], fmax=FMAX)
axes[1].set_title("Reconstructed Spectrogram", fontsize=16)
axes[1].set_ylabel('Frequency (Hz)', fontsize=16)
axes[1].set_xlabel('Time (s)', fontsize=16)

# --- Colorbar ---
fig.colorbar(img, ax=axes.tolist(), format='%+2.0f dB')

plt.savefig("reconstruction_repeated_labels.png")
print("Plot saved to reconstruction_repeated_labels.png")
plt.show()

# --- Part 2: Auditory Evaluation ---
print("\n--- Generating Audio for Comparison ---")

reconstructed_power = librosa.db_to_power(reconstructed_db)
reconstructed_waveform = librosa.feature.inverse.mel_to_audio(
    reconstructed_power, 
    sr=SR,
    n_fft=N_FFT,                         
    hop_length=HOP_LENGTH,                
    length=len(original_waveform)       
)

# play/save the audio
original_save_path = "original_audio.wav"
recon_save_path = "reconstructed_audio.wav"
sf.write(original_save_path, original_waveform, SR)
sf.write(recon_save_path, reconstructed_waveform, SR)
print(f"\nAudio successfully saved!")
print(f"Original: {original_save_path}")
print(f"Reconstructed: {recon_save_path}")

# --- Part 3: Difference Analysis ---
print("\nGenerating the separate difference plot...")

# 1. Calculate the Difference
difference_db = original_db - reconstructed_db

# 2. Create a new figure and a single axis
fig_diff, ax_diff = plt.subplots(1, 1, figsize=(12, 7), layout='constrained')
fig_diff.suptitle('Spectrogram Difference (Original - Reconstructed)', fontsize=20)

# 3. Find max absolute difference for symmetrical color scaling
max_diff = np.max(np.abs(difference_db))

# 4. Display the difference spectrogram
img_diff = librosa.display.specshow(difference_db, sr=SR, x_axis='time', y_axis='mel',
                                    ax=ax_diff, fmax=FMAX, cmap='twilight_shifted',
                                    vmin=-max_diff, vmax=max_diff)

# 5. Set labels
ax_diff.set_ylabel('Frequency (Hz)', fontsize=16)  
ax_diff.set_xlabel('Time (s)', fontsize=16)

# 6. Colorbar 
fig_diff.colorbar(img_diff, ax=ax_diff, format='%+2.0f dB', label='Difference (dB)')

# 7. Save and show the new plot
plt.savefig("difference_spectrogram.png")
print("Difference plot saved to difference_spectrogram.png")
plt.show()