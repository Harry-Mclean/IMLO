import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn


class IMLONetwork(nn.Module):
    # Set up Network
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':
    # Initialises Model
    model = IMLONetwork()

    # sets transforms for images
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Downloads dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Loads dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
