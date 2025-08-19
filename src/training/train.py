# train.py

import os
import datetime
import argparse

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchsummary import summary

from src.model.model_arch import autoencoder
from src.utils import file_paths

# ---------------------------------------------------
# Default configuration
# ---------------------------------------------------
paths = file_paths()

DATA_DIR = paths["DATA_DIR"]
MODEL_DIR = paths["MODEL_DIR"]
PLOT_DIR = paths["PLOT_DIR"]

DEFAULT_WEIGHTS_DIR = MODEL_DIR
DEFAULT_PLOTS_DIR = PLOT_DIR

DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 256
DEFAULT_BOTTLENECK_SIZE = 32


# ------------------------------
# Utility Functions
# ------------------------------
def init_weights(m):
    """Initialize weights for Linear layers using Xavier uniform."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# ------------------------------
# Training Loop
# ------------------------------
def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device, save_path=None, plot_file=None):
    print('Starting training...\n')
    model.train()
    losses_train = []

    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}/{n_epochs}")
        loss_train = 0.0

        for imgs, _ in train_loader:
            # Flatten images: (batch_size, 1, 28, 28) -> (batch_size, 784)
            imgs = imgs.view(imgs.shape[0], -1).to(device)

            # Forward pass
            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        # Step the scheduler based on total loss
        scheduler.step(loss_train)

        # Average loss per epoch
        epoch_loss = loss_train / len(train_loader)
        losses_train.append(epoch_loss)

        print(f"{datetime.datetime.now().strftime("%I:%M:%S %p")} - Training loss: {epoch_loss:.6f}")

        # Save model after each epoch
        if save_path:
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to: {save_path}")

        # Save plot of loss
        if plot_file:
            plt.figure(figsize=(10, 6))
            plt.plot(losses_train, label='Train Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(plot_file)
            print(f"Plot saved to: {plot_file}\n")


# ------------------------------
# Command-line Argument Parsing
# ------------------------------
def parse_args():
    # CLI argument parser
    parser = argparse.ArgumentParser(description="Train an MNIST Autoencoder")

    parser.add_argument('-e', type=int, default=DEFAULT_EPOCHS, help='Number of epochs (default: 30)')
    parser.add_argument('-b', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size (default: 256)')
    parser.add_argument('-z', type=int, default=DEFAULT_BOTTLENECK_SIZE, help='Bottleneck size (default: 32)')

    return parser.parse_args()


# ------------------------------
# Main
# ------------------------------
def main():
    # Parse command line arguments
    args = parse_args()

    save_file = os.path.join(DEFAULT_WEIGHTS_DIR, f'weights_E{args.e}_B{args.b}_BN{args.z}.pth')
    plot_file = os.path.join(DEFAULT_PLOTS_DIR, f'loss_E{args.e}_B{args.b}_BN{args.z}.png')    

    # Display config
    print(f"\033[1mRunning main with the following parameters:\033[0m")
    print(f"Epochs: {args.e}")
    print(f"Batch size: {args.b}")
    print(f"Bottleneck size: {args.z}\n")
    print(f"Save path: {save_file}")
    print(f"Plot file: {plot_file}\n")

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Model initialization
    N_input = 28 * 28
    model = autoencoder(N_input=N_input, N_bottleneck=args.z, N_output=N_input)
    model.to(device)
    model.apply(init_weights)
    summary(model, model.input_shape)

    # Data loaders
    transform = transforms.ToTensor()
    train_set = MNIST(DATA_DIR, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.b, shuffle=True)

    # Optimizer, scheduler, loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_fn = nn.MSELoss()

    # Train the model
    train(
        n_epochs=args.e,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        scheduler=scheduler,
        device=device,
        save_path=save_file,
        plot_file=plot_file
    )


if __name__ == '__main__':
    main()