import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn


class IMLONetwork(nn.Module):
    # Set up Network
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.layer2 = nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)  # flatten data
        x = self.layer2(x)
        return x

