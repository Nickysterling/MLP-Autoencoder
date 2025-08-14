# img_interpolate.py

import torch
import matplotlib.pyplot as plt
import numpy as np

def interpolate_images(model, dataset, device, n_steps=8):
    # Get user input for the two image indices
    idx1 = int(input("Enter the index of the first image (0-59999): "))
    idx2 = int(input("Enter the index of the second image (0-59999): "))
    
    # Get the images and flatten them for the autoencoder
    img1 = dataset.data[idx1].float() / 255.0
    img1 = img1.view(1, -1).to(device)

    img2 = dataset.data[idx2].float() / 255.0
    img2 = img2.view(1, -1).to(device)

    # Get bottleneck representations
    with torch.no_grad():
        bottleneck1 = model.encode(img1)
        bottleneck2 = model.encode(img2)

    # Create interpolated bottleneck vectors
    interpolated_images = []
    for i in range(n_steps):
        alpha = i / (n_steps - 1)
        interpolated_bottleneck = (1 - alpha) * bottleneck1 + alpha * bottleneck2

        # Decode the interpolated bottleneck back to an image
        with torch.no_grad():
            output = model.decode(interpolated_bottleneck)
            interpolated_images.append(output.view(28, 28).cpu().numpy())

    # Plot the interpolated images
    f, axarr = plt.subplots(1, n_steps, figsize=(15, 2))
    f.subplots_adjust(wspace=0.5)

    for i in range(n_steps):
        axarr[i].imshow(interpolated_images[i], cmap='gray')
    plt.show()
