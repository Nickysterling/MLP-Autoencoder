# img_reconstruct.py

import torch
import matplotlib.pyplot as plt

def display_image(model, dataset, device):
    
    index = int(input("Enter an image index (0-59999): "))

    # Fetch the image and label
    img = dataset.data[index].float() / 255.0  # Convert to float and normalize in one step
    label = dataset.targets[index].item()

    # Prepare for the model
    img = img.to(device=device)  # Transfer to device
    img = img.view(1, -1)  # Flatten the image for the model input

    # Model inference
    with torch.no_grad():  # Ensure no gradients are computed
        output = model(img)

    # Reshape the output to display it as an image
    output = output.view(28, 28)

    # Display the images
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(img.view(28, 28).cpu(), cmap='gray')
    plt.title(f'Original Label: {label}')

    f.add_subplot(1, 2, 2)
    plt.imshow(output.cpu(), cmap='gray')
    plt.title('Reconstructed Image')
    plt.show()