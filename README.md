# Generative AI Framework for Creating a Music Dataset with Quantifiable Acoustic and Cognitive Features

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

This project addresses the lack of controlled, unfamiliar musical stimuli in music cognition research by developing a dataset of AI-generated music annotated with acoustic and emotional features. We explored two approaches: (1) building a custom Latent Diffusion Model (LDM), and (2) evaluating five commercial text-to-music systems through subjective and objective analysis. Although our LDM failed to produce usable audio, it provided insight into architectural challenges in modern generative systems. The comparative study—combining listener feedback and Music Information Retrieval (MIR) analysis—identified Suno as the optimal model, offering a strong balance between musical complexity and user appeal. This project establishes a scalable music generation pipeline, laying the groundwork for an annotated dataset that supports experimental research in music perception.

**Keywords:** Generative AI, Music Cognition, Text-to-Music, Music Dataset, Latent Diffusion Model, Music Information Retrieval

---

## Table of Contents

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Latent Diffusion Model (LDM)](#latent-diffusion-model-ldm)
  - [Music Information Retrieval (MIR)](#music-information-retrieval-mir)
  - [Subjective Analysis](#subjective-analysis)
- [License](#license)
- [Contact](#contact)

---

## About The Project

This repository contains the code and resources for the project "Generative AI Framework for Creating a Music Dataset with Quantifiable Acoustic and Cognitive Features". The primary goal is to create a database of AI-generated music for music cognition research, addressing the need for diverse and controlled musical stimuli.

This project is divided into two main parts:
1.  **Custom Model Development:** The design and implementation of a Text-to-Music Latent Diffusion Model (LDM).
2.  **Commercial Model Analysis:** A comparative study of five commercial text-to-music platforms (Suno, Udio, EasyMusic, Soundraw, and Aiva) to select the most suitable generative engine for the dataset.

---

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

This project uses `conda` for environment management. You will need to have Anaconda or Miniconda installed on your system.

1.  **Create and activate the conda environment:**
    All the necessary libraries are listed in the `environment.yml` file. Create the conda environment using the following command:
    ```sh
    conda env create -f environment.yml
    ```
2.  **Activate the new environment:**
    ```sh
    conda activate music-ai-env
    ```

### Installation

1.  **Clone the repo:**
    ```sh
    git clone [https://github.com/polvime/GenAIDataset.git](https://github.com/polvime/GenAIDataset.git)
    ```
2.  **Navigate to the project directory:**
    ```sh
    cd GenAIDataset
    ```

---

## Project Structure

The repository is organized as follows:


* **`LDM/`**: Source code for the Latent Diffusion Model.
* **`MIR/`**: Scripts for Music Information Retrieval analysis.
* **`Subjective/`**: Data and analysis from the subjective assessment study.
* **`environment.yml`**: Conda environment file with all dependencies.
* **`README.md`**: This README file.

---

## Usage

This section explains how to use the scripts in the different project folders.

### Latent Diffusion Model (LDM)

The `LDM/` directory contains the implementation of the text-to-music Latent Diffusion Model.

-   `AutoEncoder.py` & `GAN.py`: Implement and train the Variational Autoencoder and Generative Adversarial Network.
-   `UNet.py`: Contains the UNet architecture and training for the denoising process.
-   `inference.py`: Script to generate music from a text prompt using the trained model.
-   `test_autoencoder.py`: Used for testing the autoencoder's reconstructions.
-   `reconstruction_comparison`: Script to create a comparative plot of the reconstructions from the AE and GAN.
-   `plot_losses.py`: Used to visualize the training and validation losses.

The `Runs&Weights` and `Reconstructions` folders contain the outputs and saved models from the training process.

### Music Information Retrieval (MIR)

The `MIR/` directory contains scripts for extracting and analyzing acoustic features from the generated music tracks.

-   `MIRComputation.m`: A MATLAB script using the MIRToolbox to compute acoustic features from audio files.
-   `Acoustic_Visualisation.html`: An HTML file to visualize the acoustic features with significant difference across models. Programmed in R.
-   `Acoustic_Sig`: An HTML file to obtain a list of the acoustic features with significant difference across models along with their p-value. Programmed in R.

### Subjective Analysis

The `Subjective/` directory holds the results and plots from the subjective assessment of the commercial models. This includes participant ratings on liking, complexity, perceived humanity, and curiosity. The `.png` and `.pdf` files contain the final plots and statistical significance reports from the analysis.

---

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

---

## Contact

Pol Vierge - pol.vierge@autonoma.cat

Project Link: [https://github.com/polvime/GenAIDataset.git](https://github.com/polvime/GenAIDataset.git)

---

