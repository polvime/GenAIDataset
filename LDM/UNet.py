# --- Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import librosa
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import json

from datasets import load_dataset
from transformers import CLIPProcessor, CLIPTextModel
from diffusers import DDPMScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

# --- Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
torch.backends.cudnn.benchmark = True


# --- Configuration ---
# Bare in mind that the data loading and paths may need to be adjusted based on your local setup.
config = {
    "batch_size": 64,
    "epochs": 50,
    "learning_rate": 1e-4,
    "unet_in_channels": 4, # Must match autoencoder's LATENT_CHANNELS
    "unet_out_channels": 4, # Must match autoencoder's LATENT_CHANNELS
    "num_timesteps": 1000, # Diffusion steps
    "n_mels": 128,          # Must match autoencoder's N_MELS
    "cut_secs": 10,         
    "text_model_id": "openai/clip-vit-base-patch32",
    "data_subset_percent": 1.0, # Use 10% of the full dataset for training tests, it's way too big otherwise (+2 days)
    "autoencoder_path": "Runs&Weights/GAN/autoencoder.pth",
    "datashare_root": "MusicBench/datashare",
    "save_dir": "UNet_runs"
}

# --- 1. Autoencoder Definition (from GAN.py) ---
class ResnetBlockAE(nn.Module):
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
            ResnetBlockAE(64, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Downsample
            ResnetBlockAE(128, 128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # Downsample
            ResnetBlockAE(256, 256),
            nn.Conv2d(256, latent_channels, kernel_size=3, stride=2, padding=1) # Final compression
        )

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, kernel_size=3, stride=1, padding=1),
            ResnetBlockAE(256, 256),
            # Upsample Block 1
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            ResnetBlockAE(128, 128),
            # Upsample Block 2
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            ResnetBlockAE(64, 64),
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
    
# --- 2. UNet Building Blocks (with Conditioning) ---
class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]; return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads; self.head_dim = query_dim // n_heads
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)
    def forward(self, x, context):
        b, c, h, w = x.shape; x_in = x
        x = x.view(b, c, -1).permute(0, 2, 1)
        q = self.to_q(x)
        k = self.to_k(context); v = self.to_v(context)
        q = q.view(b, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(b, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(b, -1, c)
        return x_in + self.to_out(attn_output).permute(0, 2, 1).view(b, c, h, w)

class ResnetBlockUNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2))
        self.block1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.block2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm1 = nn.GroupNorm(8, out_channels); self.norm2 = nn.GroupNorm(8, out_channels)
    def forward(self, x, time_emb):
        time_emb = self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = time_emb.chunk(2, dim=1)
        h = self.norm1(self.block1(F.silu(x))); h = h * (scale + 1) + shift
        h = self.norm2(self.block2(F.silu(h))); return h + self.res_conv(x)

class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, context_dim):
        super().__init__()
        self.res = ResnetBlockUNet(in_channels, out_channels, time_emb_dim)
        self.attn = CrossAttention(out_channels, context_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
    def forward(self, x, time_emb, context):
        x = self.res(x, time_emb)
        x = self.attn(x, context)
        x = self.downsample(x)
        return x

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, context_dim):
        super().__init__()

        self.res = ResnetBlockUNet(in_channels + out_channels, out_channels, time_emb_dim)
        self.attn = CrossAttention(out_channels, context_dim)

    def forward(self, x, skip_x, time_emb, context):
        x = F.interpolate(x, size=skip_x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip_x, x], dim=1)
        x = self.res(x, time_emb)
        x = self.attn(x, context)
        return x


# --- 3. The Main Conditioned UNet Model ---
class ConditionedUNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=128, context_dim=768):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPositionalEmbeddings(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim * 4), nn.SiLU(), nn.Linear(time_emb_dim * 4, time_emb_dim))
        self.init_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down1 = UNetDownBlock(64, 128, time_emb_dim, context_dim)
        self.down2 = UNetDownBlock(128, 256, time_emb_dim, context_dim)
        self.bot_res1 = ResnetBlockUNet(256, 512, time_emb_dim); self.bot_attn = CrossAttention(512, context_dim); self.bot_res2 = ResnetBlockUNet(512, 512, time_emb_dim)
        self.up1 = UNetUpBlock(512, 256, time_emb_dim, context_dim)
        self.up2 = UNetUpBlock(256, 128, time_emb_dim, context_dim)
        self.up3 = UNetUpBlock(128, 64, time_emb_dim, context_dim)
        self.output_conv = nn.Sequential(nn.GroupNorm(8, 64), nn.SiLU(), nn.Conv2d(64, out_channels, 3, padding=1))
    def forward(self, x, time, context):
        t = self.time_mlp(time)
        s1 = self.init_conv(x); s2 = self.down1(s1, t, context); s3 = self.down2(s2, t, context)
        b = self.bot_res1(s3, t); b = self.bot_attn(b, context); b = self.bot_res2(b, t)
        u1 = self.up1(b, s3, t, context); u2 = self.up2(u1, s2, t, context); u3 = self.up3(u2, s1, t, context)
        return self.output_conv(u3)


# --- 4. Dataset and Dataloading ---
class MusicDataset(Dataset):
    def __init__(self, hf_dataset, audio_files_root, sr=22050, n_mels=128, cut_secs=10):
        self.dataset = hf_dataset; self.audio_root = Path(audio_files_root)
        self.sr = sr; self.n_mels = n_mels; self.cut_len = sr * cut_secs
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        try:
            mix, _ = librosa.load(self.audio_root / item["location"], sr=self.sr, mono=True)
            if len(mix) > self.cut_len: start = (len(mix) - self.cut_len) // 2; mix = mix[start : start + self.cut_len]
            else: mix = np.pad(mix, (0, self.cut_len - len(mix)), 'constant')
            if np.max(np.abs(mix)) == 0: return self.__getitem__((idx + 1) % len(self))
            mix_mel = librosa.feature.melspectrogram(y=mix, sr=self.sr, n_mels=self.n_mels)
            mix_db = librosa.power_to_db(mix_mel, ref=np.max)
            mix_db = (mix_db - mix_db.min()) / (mix_db.max() - mix_db.min() + 1e-6)
            caption = f"{item.get('main_caption', '')} {item.get('prompt_key', '')} {item.get('prompt_bpm', '')}"
            return {"spectrogram": torch.tensor(mix_db, dtype=torch.float32), "text": caption}
        except Exception: return self.__getitem__((idx + 1) % len(self))

def collate_unet(batch):
    batch = [item for item in batch if item is not None];
    if not batch: return None
    spectrograms = torch.stack([item['spectrogram'] for item in batch])
    texts = [item['text'] for item in batch]
    return {"spectrograms": spectrograms, "texts": texts}


# --- 5. Training ---
def main():
    print("Loading MusicBench dataset...")
    full_ds = load_dataset("amaai-lab/MusicBench", split="train")

    # --- Create a subset of the dataset ---
    subset_percent = config.get("data_subset_percent", 1.0) # Default to 100%
    if subset_percent < 1.0:
        num_samples = int(len(full_ds) * subset_percent)
        ds_subset = full_ds.select(range(num_samples)) 
        print(f"Using a subset of {num_samples} samples ({subset_percent*100:.0f}% of the full training set).")
    else:
        ds_subset = full_ds
        print("Using the full training dataset.")
    print(f"Example from dataset: {ds_subset[0]}")
    unet_dataset = MusicDataset(ds_subset, config["datashare_root"], n_mels=config["n_mels"], cut_secs=config["cut_secs"])
    print(f"Loaded {len(unet_dataset)} samples from MusicBench dataset.")
    unet_loader = DataLoader(unet_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_unet, num_workers=4, pin_memory=True)

    print("Loading validation data...")
    test_ds = load_dataset("amaai-lab/MusicBench", split="test") 

    val_dataset = MusicDataset(test_ds, config["datashare_root"], n_mels=config["n_mels"], cut_secs=config["cut_secs"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_unet, num_workers=4, pin_memory=True)

    print("Loading pre-trained Autoencoder...")
    if not os.path.exists(config["autoencoder_path"]):
        print(f"FATAL: Autoencoder weights not found at {config['autoencoder_path']}"); return
    autoencoder = Autoencoder(latent_channels=config["unet_in_channels"]).to(device)
    autoencoder.load_state_dict(torch.load(config["autoencoder_path"], map_location=device))
    autoencoder.eval(); [p.requires_grad_(False) for p in autoencoder.parameters()]
    print("Pre-trained Autoencoder loaded.")

    print("Loading CLIP Text Encoder...")
    text_processor = CLIPProcessor.from_pretrained(config["text_model_id"])
    text_model = CLIPTextModel.from_pretrained(config["text_model_id"]).to(device)
    text_model.eval(); [p.requires_grad_(False) for p in text_model.parameters()]
    print("CLIP Text Encoder loaded.")

    print("Initializing Conditioned UNet...")
    unet = ConditionedUNet(in_channels=config["unet_in_channels"], out_channels=config["unet_out_channels"], context_dim=512).to(device)
    print(f"ConditionedUNet initialized with {sum(p.numel() for p in unet.parameters() if p.requires_grad):,} parameters.")

    #create a run folder in the save_dir with the current date and time
    run_name = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config["save_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory created: {run_dir}")

    print("Starting UNet training...")
    scheduler = DDPMScheduler(num_train_timesteps = config["num_timesteps"], beta_schedule="squaredcos_cap_v2")
    optimizer = AdamW(unet.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss(); scaler = GradScaler('cuda')

    # --- Learning rate scheduler ---
    scheduler_lr = ReduceLROnPlateau(
        optimizer,
        mode='min',      
        factor=0.2,    
        patience=5,  
    )

    config_path = os.path.join(run_dir, "config_and_architecture.txt")
    with open(config_path, 'w') as f:
        f.write("--- Training Configuration ---\n")
        json.dump(config, f, indent=4)
        f.write("\n\n--- AE Architecture ---\n")
        f.write(str(autoencoder))
        f.write("\n\n--- UNet Architecture ---\n")
        f.write(str(unet))
    print(f"Configuration and architecture saved to {config_path}")

    train_loss_history = []
    val_loss_history = []

    for epoch in range(config["epochs"]):
        unet.train()
        total_train_loss = 0
        progress_bar = tqdm(unet_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch in progress_bar:
            if batch is None: continue
            spectrograms = batch["spectrograms"].to(device, non_blocking=True); texts = batch["texts"]
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                with torch.no_grad():
                    text_inputs = text_processor(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=77).to(device)
                    text_embeddings = text_model(**text_inputs).last_hidden_state
                    true_latents = autoencoder.encoder(spectrograms.unsqueeze(1))
                noise = torch.randn_like(true_latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (true_latents.shape[0],), device=device)
                noisy_latents = scheduler.add_noise(true_latents, noise, timesteps)
                predicted_noise = unet(noisy_latents, timesteps, text_embeddings)
                loss = loss_fn(predicted_noise, noise)
           
            scaler.scale(loss).backward()
            # --- Clip the gradients to prevent them from exploding ---
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            progress_bar.set_postfix(loss=f"{loss.item():.5f}")
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(unet_loader)
        train_loss_history.append(avg_train_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_train_loss:.5f}")
        
        # Save model checkpoint
        os.makedirs(run_dir, exist_ok=True)
        checkpoints_path = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoints_path, exist_ok=True)
        torch.save(unet.state_dict(), os.path.join(checkpoints_path, f"unet_epoch_{epoch+1}.pth"))
        print(f"Checkpoint saved for epoch {epoch+1} at {checkpoints_path}")

        # --- VALIDATION PHASE ---
        unet.eval() 
        total_val_loss = 0
        with torch.no_grad(): 
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Validation]")
            for batch in val_progress_bar:
                if batch is None: continue
                spectrograms = batch["spectrograms"].to(device, non_blocking=True); texts = batch["texts"]

                text_inputs = text_processor(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=77).to(device)
                text_embeddings = text_model(**text_inputs).last_hidden_state
                true_latents = autoencoder.encoder(spectrograms.unsqueeze(1))

                noise = torch.randn_like(true_latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (true_latents.shape[0],), device=device)
                noisy_latents = scheduler.add_noise(true_latents, noise, timesteps)

                predicted_noise = unet(noisy_latents, timesteps, text_embeddings)
                loss = loss_fn(predicted_noise, noise)

                val_progress_bar.set_postfix(loss=f"{loss.item():.5f}")
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        print(f"Epoch {epoch+1} Average Validation Loss: {avg_val_loss:.5f}")
        scheduler_lr.step(avg_val_loss)
    
    # Final save
    print("UNet training complete. Saving artifacts...")

    # 1. Save model weights
    weights_path = os.path.join(run_dir, "autoencoder_weights.pth")
    torch.save(unet.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}")

    # 2. Save loss history
    train_loss_path = os.path.join(run_dir, "train_loss_history.json")
    val_loss_path = os.path.join(run_dir, "val_loss_history.json")
    with open(train_loss_path, 'w') as f:
        json.dump(train_loss_history, f)
    with open(val_loss_path, 'w') as f:
        json.dump(val_loss_history, f)
    print(f"Loss histories saved to {run_dir}")

    # 3. Save plot
    plot_path = os.path.join(run_dir, "loss_plot.png")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, config["epochs"] + 1), train_loss_history, marker='o', label='Training Loss')
    plt.plot(range(1, config["epochs"] + 1), val_loss_history, marker='x', label='Test Loss')
    plt.title(f"Training Loss - Run {run_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    tick_spacing = max(1, config["epochs"] // 10) 
    plt.xticks(np.arange(1, config["epochs"] + 1, tick_spacing))
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Loss plot saved to {plot_path}")

if __name__ == '__main__':
    main()