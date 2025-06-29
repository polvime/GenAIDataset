import json
import matplotlib.pyplot as plt

def plot_training_histories(ae_train_file, ae_val_file, gan_history_file, unet_train_file, unet_val_file, output_filename="all_training_losses_horizontal.png"):
    """
    Loads training loss data and creates a 3-panel horizontal plot comparing
    the training histories of an AE, GAN-AE, and U-Net, with independent axes.

    Args:
        ae_train_file (str): Path to the AE training loss JSON.
        ae_val_file (str): Path to the AE validation loss JSON.
        gan_history_file (str): Path to the GAN training history JSON.
        unet_train_file (str): Path to the U-Net training loss JSON.
        unet_val_file (str): Path to the U-Net validation loss JSON.
        output_filename (str): Name of the file to save the plot.
    """
    try:
        # --- 1. Load all data from JSON files ---
        print("Loading loss history files...")
        with open(ae_train_file, 'r') as f:
            ae_train_loss = json.load(f)
        with open(ae_val_file, 'r') as f:
            ae_val_loss = json.load(f)
        with open(gan_history_file, 'r') as f:
            gan_history = json.load(f)
        with open(unet_train_file, 'r') as f:
            unet_train_loss = json.load(f)
        with open(unet_val_file, 'r') as f:
            unet_val_loss = json.load(f)
        print("All files loaded successfully.")

        # --- 2. Create the figure and subplots ---
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), layout='constrained')
        fig.suptitle('Comparative Model Training History', fontsize=22)

        # --- 3. Plot 1: Standard Autoencoder ---
        ax1 = axes[0]
        ax1.plot(ae_train_loss, label='Train Loss', color='#AE2D68', linewidth=2)
        ax1.plot(ae_val_loss, label='Validation Loss', color='#E8761E', linestyle='--', linewidth=2)
        ax1.set_title('Autoencoder Loss', fontsize=18)
        ax1.set_ylabel('Loss', fontsize=14)
        ax1.set_xlabel('Epochs', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.set_xlim(0, len(ae_train_loss) - 1)

        # --- 4. Plot 2: GAN Autoencoder ---
        ax2 = axes[1]
        
        g_loss_scaled = [loss / 10.0 for loss in gan_history.get("G_loss", [])]
        d_loss = gan_history.get("D_loss", [])
        val_recon_loss = [loss * 10.0 for loss in gan_history.get("val_recon_loss", [])]
        
        ax2.plot(g_loss_scaled, label='Generator Loss (G_loss)', color='#AE2D68', linewidth=2) 
        ax2.plot(d_loss, label='Discriminator Loss (D_loss)', color='#2ca02c', linewidth=2)
        ax2.plot(val_recon_loss, label='Validation Reconstruction Loss', color='#E8761E', linestyle='--', linewidth=2)
        ax2.set_title('GAN Autoencoder Losses', fontsize=18)
        ax2.set_ylabel('Loss', fontsize=14)
        ax2.set_xlabel('Epochs', fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.set_xlim(0, len(d_loss) - 1)


        # --- 5. Plot 3: U-Net ---
        ax3 = axes[2]
        ax3.plot(unet_train_loss, label='Train Loss', color='#AE2D68', linewidth=2)
        ax3.plot(unet_val_loss, label='Validation Loss', color="#E8761E", linestyle='--', linewidth=2)
        ax3.set_title('U-Net Loss', fontsize=18)
        ax3.set_ylabel('Loss', fontsize=14)
        ax3.set_xlabel('Epochs', fontsize=14)
        ax3.legend(fontsize=12)
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.set_xlim(0, len(unet_train_loss) - 1)

        # --- 6. Save the final plot ---
        plt.savefig(output_filename, dpi=300)
        print(f"Plot successfully saved to {output_filename}")
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: Could not find the file {e.filename}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Bare in mind that paths may need to be adjusted based on your local setup.
plot_training_histories(
    ae_train_file='training_runs/20250617_104258/train_loss_history.json',
    ae_val_file='training_runs/20250617_104258/test_loss_history.json',
    gan_history_file='GAN_runs/20250624_173138/training_history.json',
    unet_train_file='UNet_runs/20250621_170033/train_loss_history.json',
    unet_val_file='UNet_runs/20250621_170033/val_loss_history.json'
)