# MNIST Autoencoder with PyTorch

A PyTorch MLP autoencoder that compresses and reconstructs handwritten digits, demonstrating representation learning and latent space exploration.

## 1. Overview

This project implements a fully-connected (MLP) autoencoder for the MNIST handwritten digits dataset. The model learns to compress (encode) and reconstruct (decode) images of handwritten digits.

It includes interactive scripts for exploring different tasks:

- **Dataset Visualization** – preview MNIST samples.
- **Image Reconstruction** – view original and reconstructed digits.
- **Image Denoising** – remove Gaussian noise from images.
- **Image Interpolation** – smoothly transition between two different digits in latent space.

The project is written in **PyTorch** and is designed to be modular and easy to extend.

## 2. Key Features

* Fully-connected (MLP) autoencoder with configurable bottleneck size.
* Interactive Python scripts for visualization, reconstruction, denoising, and interpolation.
* Designed to run on CPU or GPU.
* Modular architecture for easy adaptation to other datasets.
* Written in PyTorch, showcasing deep learning fundamentals and good coding practices.

## 3. Autoencoder Architecture

Implemented in `src/model/model_arch.py`:

- Input layer: `784` units (flattened 28×28 MNIST images)
- Bottleneck layer: configurable size (default `8`)
- Symmetrical decoder to reconstruct original images
- Activation functions:
  - **ReLU** in hidden layers
  - **Sigmoid** in output layer

## 4. Project Structure

```
├── archive/
├── data/
├── documentation/
├── outputs/
├── src/
│ └── model/
│   └── model_arch.py
│ └── operation/
│   └── img_denoise.py
│   └── img_interpolate.py
│   └── img_reconstruct.py
│ └── training/
│   └── train.py
│ └── main.py
│ └── utils.py
│ └── visualize_dataset.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 5. Installation

**1. Clone the repository**

```
git clone https://github.com/Nickysterling/MLP-Autoencoder.git
cd MLP-Autoencoder
```

**2. Create a virtual environment**

```
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows
```

**3. Install dependencies (GPU Users)**

For GPU users with CUDA 12.6:

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

**4.** **Install dependencies (CPU Users**)

For CPU users or if you encounter errors, replace `torch` and `torchvision` in `requirements.txt` with CPU wheels:

```
torch==2.8.0+cpu
torchvision==0.23.0+cpu
```

Then install:

```
pip install -r requirements.txt
```

## 6. Training Instructions

Run `train.py` to train the autoencoder on MNIST:

```
cd MLP-Autoencoder
python -m src.training.train -e 30 -b 256 -z 32 
```

Arguments:

* `-e` : Number of training epochs (default: 30)
* `-b` : Batch size (default: 256)
* `-z` : Bottleneck size (default: 32)

## 7. Usage Instructions

Once training is finished, you can test the autoencoder using `main.py`.

**1. Navigate to the project root directory**

```
cd MLP-Autoencoder
```

**2. Run `main.py` and provide your trained model weights**

If you want to use the example weights provided in the repo use this command:

```
python -m src.main -w archive\model_weights\model_weights.pth
```

If you trained your own model with `train.py`, use this command and replace `weights.pth` with the name of your saved file:

```
python -m src.main -w outputs\model_weights\model_weights.pth
```

When it runs, you will be prompted to select MNIST image indices for each task. Each index corresponds to an image in the MNIST dataset. You can view individual images using the `visualize_dataset.py` script.

```
python -m src.visualize_dataset
```

## 8. Example Outputs

### 8.1. Sample Images

**Left Index: 25, Right Index: 50**

| ![Index 25](https://github.com/Nickysterling/mlp_autoencoder/blob/main/documentation/img/idx_25.png?raw=true "Index 25") | ![Index 50](https://github.com/Nickysterling/mlp_autoencoder/blob/main/documentation/img/idx_50.png?raw=true "Index 50") |
| ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |

### 8.2. Image Reconstruction

| ![Reconstruction Example](https://github.com/Nickysterling/mlp_autoencoder/blob/main/documentation/img/idx_25_reconstructed.png?raw=true "Index 25 Reconstruction") | ![Reconstruction Example](https://github.com/Nickysterling/mlp_autoencoder/blob/main/documentation/img/idx_50_reconstructed.png?raw=true "Index 50 Reconstruction") |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |

### 8.3. Image Denoising

| ![Denoising Example](https://github.com/Nickysterling/mlp_autoencoder/blob/main/documentation/img/idx_25_denoise.png?raw=true "Index 25 Denoising") | ![Denoising Example](https://github.com/Nickysterling/mlp_autoencoder/blob/main/documentation/img/idx_50_denoise.png?raw=true "Index 50 Denoising") |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |

### 8.4. Image Interpolation

![Interpolation Example](https://github.com/Nickysterling/mlp_autoencoder/blob/main/documentation/img/interpolate.png?raw=true "Interpolation")

## 9. Skills and Technologies

**Languages & Libraries**

* Python, NumPy, Matplotlib
* PyTorch (MLP autoencoders, training loops, GPU support)

**Machine Learning / Deep Learning Concepts**

* Autoencoders (encoding, decoding, bottleneck representation)
* Representation learning & latent space exploration
* Image reconstruction, denoising, interpolation

## 10. Notes

* The model can run on **CPU** or **GPU** (if available).
* Latent space size (`N_bottleneck`) affects compression and reconstruction quality.
* You can adapt this code for other datasets by adjusting input size and transforms.
