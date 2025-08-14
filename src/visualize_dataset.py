# visualize_dataset.py

import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

# .compose allows to chain multiple transformation, .ToTensor changes the image to a PyTorch tensor object
train_transform = transforms.Compose([transforms.ToTensor()])

# train_set is a dataset object that contains the training images and labels from the MNIST dataset
train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)

# Display the image with its label
def display_image(index):
    image = train_set.data[index]
    label = train_set.targets[index]
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.show()

index = int(input("Enter an image index (0-59999): "))
display_image(index)