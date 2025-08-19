# visualize_dataset.py

import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from src.utils import file_paths

paths = file_paths()

DATA_DIR = paths["DATA_DIR"]

# Display the image with its label
def display_image(index, dataset, dataset_name="train"):
    image = dataset.data[index]
    label = dataset.targets[index]
    plt.imshow(image, cmap='gray')
    plt.title(f'{dataset_name} - Label: {label}')
    plt.show()

if __name__ == "__main__":
    # Change images to tensor
    transform = transforms.Compose([transforms.ToTensor()])

    # Create train/val sets
    train_set = MNIST(DATA_DIR, train=True, download=True, transform=transform)
    val_set = MNIST(DATA_DIR, train=False, download=True, transform=transform)

    while True:
        # Let user pick dataset
        dataset_choice = input("Choose dataset: (t)rain or (v)al? ").strip().lower()
        if dataset_choice == 't':
            dataset = train_set
            dataset_name = "Train"
            max_index = len(train_set) - 1
        elif dataset_choice == 'v':
            dataset = val_set
            dataset_name = "Validation"
            max_index = len(val_set) - 1
        else:
            print("Invalid choice, please enter 't' or 'v'.")
            continue

        # Pick index
        try:
            index = int(input(f"Enter an image index (0-{max_index}): "))
            if 0 <= index <= max_index:
                display_image(index, dataset, dataset_name)
            else:
                print("Index out of range.")
                continue
        except ValueError:
            print("Please enter a valid integer index.")
            continue

        # Continue or quit
        visualize = input("\nPress enter to continue or 'q' to quit: ")
        if visualize.lower() == 'q':
            break