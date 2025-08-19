# main.py

import argparse
import sys
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from src.model.model_arch import autoencoder
from src.operation import img_reconstruct
from src.operation import img_denoise
from src.operation import img_interpolate
from src.utils import file_paths

paths = file_paths()

DATA_DIR = paths["DATA_DIR"]

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Run the autoencoder model on MNIST data.')
    parser.add_argument('-w', required=True, help='Path to the saved model file')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Validation set
    transform = transforms.Compose([transforms.ToTensor()])
    val_set = MNIST(DATA_DIR, train=False, download=True, transform=transform)

    # Load the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    model = autoencoder(N_input=784, N_bottleneck=8, N_output=784)
    model.load_state_dict(torch.load(args.w, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Step 04
    print("\nRunning Image Reconstruction")
    img_reconstruct.display_image(model, val_set, device)

    # Step 05
    print("\nRunning Image Denoising")
    img_denoise.denoise_image(model, val_set, device)

    # Step 06
    print("\nRunning Image Interpolation")
    img_interpolate.interpolate_images(model, val_set, device, n_steps=10)

    return 0

if __name__ == '__main__':
    main()
