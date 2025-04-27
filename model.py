import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn


class IMLONetwork(nn.Module):
    # Set up Network
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.layer2 = nn.ReLU()  # apply Relu
        self.layer3 = nn.MaxPool2d(2, 2)  # Pooling
        self.layer4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.layer5 = nn.ReLU()
        self.layer6 = nn.MaxPool2d(2, 2)  # Pooling after the new convolutional layer

        self.final_layer = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)  # flatten data
        x = self.final_layer(x)
        return x
