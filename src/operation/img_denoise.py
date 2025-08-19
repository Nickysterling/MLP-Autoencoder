# img_denoise.py

import torch
import matplotlib.pyplot as plt

def denoise_image(model, dataset, device):
    
    index = int(input("Enter an image index (0-59999): "))

    # Fetch the image and label
    img = dataset.data[index].float() / 255.0  # Normalize the image
    label = dataset.targets[index].item()

    # Add noise to the image
    noisy_img = img + (0.1**0.5) * torch.randn(28, 28)

    # Prepare noisy image for the model
    noisy_img = noisy_img.to(device=device)
    noisy_img = noisy_img.view(1, -1)  # Flatten the image for the model input

    with torch.no_grad():
        output = model(noisy_img).view(28, 28)  # Denoising

    # Display the original, noisy, and reconstructed images
    f, axarr = plt.subplots(1, 3)  
    axarr[0].imshow(img.cpu(), cmap='gray')
    axarr[0].set_title(f'Original Label: {label}')

    axarr[1].imshow(noisy_img.view(28, 28).cpu(), cmap='gray')
    axarr[1].set_title('Noisy Image')

    axarr[2].imshow(output.cpu(), cmap='gray')
    axarr[2].set_title('Denoised Image')

    plt.show()
