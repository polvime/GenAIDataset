# --- Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import librosa
import numpy as np
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
# Hugging Face imports
from transformers import CLIPProcessor, CLIPTextModel
from diffusers import DDPMScheduler

# --- 1. Autoencoder Architecture (from GAN.py) ---
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
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), ResnetBlockAE(64, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), ResnetBlockAE(128, 128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), ResnetBlockAE(256, 256),
            nn.Conv2d(256, latent_channels, kernel_size=3, stride=2, padding=1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, kernel_size=3, stride=1, padding=1), ResnetBlockAE(256, 256),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), ResnetBlockAE(128, 128),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), ResnetBlockAE(64, 64),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv2d(64, 1, kernel_size=3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return F.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=False)

# --- 2. UNet Architecture (from UNet.py) ---
class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

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
        x = self.res(x, time_emb); x = self.attn(x, context); x = self.downsample(x)
        return x

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, context_dim):
        super().__init__()
        self.res = ResnetBlockUNet(in_channels + out_channels, out_channels, time_emb_dim)
        self.attn = CrossAttention(out_channels, context_dim)
    def forward(self, x, skip_x, time_emb, context):
        x = F.interpolate(x, size=skip_x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip_x, x], dim=1)
        x = self.res(x, time_emb); x = self.attn(x, context)
        return x

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


def main():
    # --- 3. Inference Configuration ---
    config = {
        # --- Paths to your trained models ---
        # Bare in mind that the data loading and paths may need to be adjusted based on your local setup.
        "autoencoder_path": "Runs&Weights/GAN/autoencoder.pth",
        "unet_path": "Runs&Weights/UNet/checkpoints/unet_epoch_75.pth",
        
        # --- Generation Parameters ---
        "prompt": "A pop piano",
        "diffusion_steps": 1000,
        "guidance_scale": 15,
        "seed": 42,

        # --- Audio & Model Parameters (should match training config) ---
        "sample_rate": 16000,
        "cut_secs": 10,
        "n_mels": 128,
        "latent_channels": 4,
        
        # --- Latent Space Shape (for 10s audio) ---
        "latent_height": 16,
        "latent_width": 40,
        
        # --- Spectrogram Denormalization ---
        "global_min_db": -80.0,
        "global_max_db": 0.0,
        
        # --- Output ---
        "output_dir": "generated_audio",
    }

    # --- 4. Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    if config["seed"] is not None:
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])

    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(exist_ok=True)

    # --- 5. Load Models ---
    print("Loading models...")
    # Autoencoder
    autoencoder = Autoencoder(latent_channels=config["latent_channels"]).to(device)
    autoencoder.load_state_dict(torch.load(config["autoencoder_path"], map_location=device))
    autoencoder.eval()
    print(f"Autoencoder loaded from {config['autoencoder_path']}")

    # UNet
    unet = ConditionedUNet(in_channels=config["latent_channels"], out_channels=config["latent_channels"], context_dim=512).to(device)
    unet.load_state_dict(torch.load(config["unet_path"], map_location=device))
    unet.eval()
    print(f"UNet loaded from {config['unet_path']}")
    
    # Text Encoder & Processor
    text_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_model.eval()
    print("CLIP models loaded.")

    # Diffusion Scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    
    # --- 6. Prepare for Generation ---
    print("Preparing text embeddings...")
    # Tokenize prompts
    prompts = ["", config["prompt"]]
    
    text_input = text_processor(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77,
    ).to(device)
    
    # Get text embeddings from CLIP
    with torch.no_grad():
        text_embeddings = text_model(**text_input).last_hidden_state
    
    uncond_embedding, cond_embedding = text_embeddings.chunk(2)
    text_embeddings = torch.cat([uncond_embedding, cond_embedding])

    # --- 7. The Denoising Loop ---
    print("Starting diffusion process...")
    # 1. Initialize random noise in the latent space
    latents = torch.randn(
        (1, config["latent_channels"], config["latent_height"], config["latent_width"]),
        device=device
    )

    # 2. Set the number of inference steps
    scheduler.set_timesteps(config["diffusion_steps"])

    # 3. The main denoising loop
    for t in tqdm(scheduler.timesteps):
        # For Classifier-Free Guidance, we process two inputs at once:
        # one conditioned on the prompt, one not.
        latent_model_input = torch.cat([latents] * 2)
        
        # We create a tensor of the correct shape and fill it with the current timestep 't'.
        time_input = torch.tensor([t] * latent_model_input.shape[0], device=device)
        
        with torch.no_grad():
            # Predict the noise for both inputs, passing the correctly shaped time_input
            noise_pred = unet(latent_model_input, time_input, context=text_embeddings)
        
        # Split the predictions
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        
        # Guidance formula
        noise_pred = noise_pred_uncond + config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)
        
        # Update the latents with the scheduler's step function
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        
    print("Diffusion complete.")

    # --- 8. Decode Latent to Spectrogram ---
    print("Decoding latents to spectrogram...")
    with torch.no_grad():
        spectrogram = autoencoder.decoder(latents).squeeze(0).squeeze(0)
    
    # --- 9. Post-Process and Save Audio ---
    print("Converting spectrogram to audio and saving...")
    # Convert to numpy and denormalize
    spec_np = spectrogram.cpu().numpy()
    min_db = config["global_min_db"]
    max_db = config["global_max_db"]
    denormalized_db = spec_np * (max_db - min_db) + min_db

    # --- Create a base filename for both audio and plot ---
    prompt_words = "_".join(config["prompt"][:50].replace("/", "_").split())
    base_filename = output_dir / f"{prompt_words}_{config['seed']}"

    # --- Plot and save the spectrogram ---
    plot_filename = base_filename.with_suffix('.png')
    if plot_filename.exists():
        plot_filename = f'{base_filename}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png'

    fig, ax = plt.subplots(figsize=(12, 6))
    img = librosa.display.specshow(
        denormalized_db,
        sr=config["sample_rate"],
        hop_length=512,  # Should match the hop_length used in mel_to_audio
        x_axis='time',
        y_axis='mel',
        ax=ax
    )
    fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Decibels (dB)')
    ax.set_title(f"Generated Mel Spectrogram\nPrompt: {config['prompt']}", wrap=True)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig) 

    # Convert from decibels to power
    power_spec = librosa.db_to_power(denormalized_db)

    # Convert Mel spectrogram to audio waveform 
    audio_waveform = librosa.feature.inverse.mel_to_audio(
        power_spec,
        sr=config["sample_rate"],
        n_fft=2048,
        hop_length=512
    )

    # Save the audio file
    audio_filename = base_filename.with_suffix('.wav')
    if audio_filename.exists():
        audio_filename = f'{base_filename}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.wav'
    
    sf.write(audio_filename, audio_waveform, config["sample_rate"])

    print("-" * 50)
    print(f"Spectrogram plot saved to: {plot_filename}")
    print(f"Successfully generated audio and saved to: {audio_filename}")
    print("-" * 50)

if __name__ == '__main__':
    main()