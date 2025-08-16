# train.py

import os
import sys
import datetime
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchsummary import summary

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder of train.py
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))  # go up to src/
PROJ_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))  # go up one more level
DATA_DIR = os.path.join(PROJ_DIR, 'data', 'mnist')
sys.path.append(SRC_DIR)

from model.model_arch import autoencoder

# ---------------------------------------------------
# Default training parameters
# ---------------------------------------------------
DEFAULT_WEIGHTS_DIR = os.path.join(BASE_DIR, 'model_weights')
DEFAULT_PLOTS_DIR = os.path.join(BASE_DIR, 'loss_plots')
os.makedirs(DEFAULT_WEIGHTS_DIR, exist_ok=True)
os.makedirs(DEFAULT_PLOTS_DIR, exist_ok=True)

DEFAULT_SAVE_FILE = os.path.join(DEFAULT_WEIGHTS_DIR, 'weights.pth')
DEFAULT_PLOT_FILE = os.path.join(DEFAULT_PLOTS_DIR, 'loss_plot.png')
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 256
DEFAULT_BOTTLENECK_SIZE = 32

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device, save_path=None, plot_file=None):
    print('Starting training...')
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

        print(f"{datetime.datetime.now()} - Training loss: {epoch_loss:.6f}")

        # Save model after each epoch
        if save_path:
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        # Save plot of loss
        if plot_file:
            plt.figure(figsize=(10, 6))
            plt.plot(losses_train, label='Train Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(plot_file)
            print(f"Loss plot saved to {plot_file}")


def init_weights(m):
    """Initialize weights for Linear layers using Xavier uniform."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def main():
    # CLI argument parser
    parser = argparse.ArgumentParser(description="Train an MNIST Autoencoder")
    parser.add_argument('-s', type=str, help='Save file name (default: weights.pth)')
    parser.add_argument('-z', type=int, help='Bottleneck size (default: 32)')
    parser.add_argument('-e', type=int, help='Number of epochs (default: 30)')
    parser.add_argument('-b', type=int, help='Batch size (default: 256)')
    parser.add_argument('-p', type=str, help='Loss plot output file name (default: plot.png)')
    args = parser.parse_args()

    # Use args or defaults
    save_file = os.path.join(DEFAULT_WEIGHTS_DIR, args.s) if args.s else DEFAULT_SAVE_FILE
    bottleneck_size = args.z if args.z else DEFAULT_BOTTLENECK_SIZE
    n_epochs = args.e if args.e else DEFAULT_EPOCHS
    batch_size = args.b if args.b else DEFAULT_BATCH_SIZE
    plot_file = os.path.join(DEFAULT_PLOTS_DIR, args.p) if args.p else DEFAULT_PLOT_FILE

    # Display config
    print(f"Bottleneck size: {bottleneck_size}")
    print(f"Epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Model save path: {save_file}")
    print(f"Plot file: {plot_file}")

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Model initialization
    N_input = 28 * 28
    model = autoencoder(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_input)
    model.to(device)
    model.apply(init_weights)
    summary(model, model.input_shape)

    # Data loaders
    transform = transforms.ToTensor()
    train_set = MNIST(DATA_DIR, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Optimizer, scheduler, loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_fn = nn.MSELoss()

    # Train the model
    train(
        n_epochs=n_epochs,
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