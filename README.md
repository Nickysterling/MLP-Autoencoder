# MNIST Autoencoder with PyTorch

## 1. Overview

This project implements a **fully-connected (MLP) autoencoder** for the MNIST handwritten digits dataset.The model learns to compress (encode) and reconstruct (decode) digit images, and includes interactive scripts for:

- **Dataset Visualization** – preview MNIST samples.
- **Image Reconstruction** – view original and reconstructed digits.
- **Image Denoising** – remove Gaussian noise from images.
- **Image Interpolation** – smoothly transition between two different digits in latent space.

The project is written in **PyTorch** and is designed to be modular and easy to extend.

## 2. Autoencoder Architecture

Implemented in `model/model_arch.py`:

- Input layer: `784` units (flattened 28×28 MNIST images)
- Bottleneck layer: configurable size (default `8`)
- Symmetrical decoder to reconstruct original images
- Activation functions:
  - **ReLU** in hidden layers
  - **Sigmoid** in output layer

## 3. Project Structure

```
├── model/
│ └── model_arch.py # Autoencoder model definition
├── train.py # Training loop and CLI arguments
├── main.py # Loads model and runs image tasks
├── img_reconstruct.py # Reconstructs and displays an image
├── img_denoise.py # Denoises a noisy MNIST image
├── img_interpolate.py # Interpolates between two MNIST digits
├── visualize_dataset.py # Displays MNIST dataset samples
└── data/ # MNIST dataset (downloaded automatically)
```

## 4. Installation

**1. Clone the repository**

```
git clone https://github.com/Nickysterling/mlp_autoencoder.git
cd mlp_autoencoder/src
```

**2. Create a virtual environment**

```
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows
```

**3. Install dependencies**

```
pip install -r requirements.txt
```

## 5. Training

Run `train.py` to train the autoencoder on MNIST:

```
python train.py -s weights.pth -z 32 -e 30 -b 256 -p loss_plot.png
```

Arguments:

* `-s` : Path to save model weights (`.pth`)
* `-z` : Bottleneck size (default: 32)
* `-e` : Number of training epochs (default: 30)
* `-b` : Batch size (default: 256)
* `-p` : Path to save training loss plot (`.png`)

## 6. Testing

After training, use `main.py` to test reconstruction, denoising, and interpolation:

```
cd mlp_autoencoder/src
python main.py -l model/weights.pth
```

Replace the `weights.pth` with the path to your trained model. When it runs, you will be prompted to select MNIST image indices for each task. Each index corresponds to an image in the MNIST dataset. You can view images using the `visualize_dataset.py` script.

## 7. Example Outputs

### 7.1. Sample Images

**Left Index: 25, Right Index: 50**

|  ![Index 25](https://github.com/Nickysterling/mlp_autoencoder/blob/main/documentation/img/idx_25.png?raw=true "Index 25") | ![Index 50](https://github.com/Nickysterling/mlp_autoencoder/blob/main/documentation/img/idx_50.png?raw=true "Index 50") |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |

### 7.2. Image Reconstruction

| ![Reconstruction Example](https://github.com/Nickysterling/mlp_autoencoder/blob/main/documentation/img/idx_25_reconstructed.png?raw=true "Index 25 Reconstruction") | ![Reconstruction Example](https://github.com/Nickysterling/mlp_autoencoder/blob/main/documentation/img/idx_50_reconstructed.png?raw=true "Index 50 Reconstruction") |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |

### 7.3. Image Denoising

| ![Denoising Example](https://github.com/Nickysterling/mlp_autoencoder/blob/main/documentation/img/idx_25_denoise.png?raw=true "Index 25 Denoising") | ![Denoising Example](https://github.com/Nickysterling/mlp_autoencoder/blob/main/documentation/img/idx_50_denoise.png?raw=true "Index 50 Denoising") |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |

### 7.4. Image Interpolation

![Interpolation Example](https://github.com/Nickysterling/mlp_autoencoder/blob/main/documentation/img/interpolate.png?raw=true "Interpolation")

## 8. Notes

* The model can run on **CPU** or **GPU** (if available).
* Latent space size (`N_bottleneck`) affects compression and reconstruction quality.
* You can adapt this code for other datasets by adjusting input size and transforms.
