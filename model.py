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
        self.layer4 = nn.BatchNorm2d(32)
        self.layer5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.MaxPool2d(2, 2)  # Pooling after the new convolutional layer
        self.layer8 = nn.BatchNorm2d(64)
        self.layer9 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Third convolution
        self.layer10 = nn.ReLU()  # Apply ReLU activation
        self.layer11 = nn.MaxPool2d(2, 2)  # Pooling after third convolution
        self.layer12 = nn.BatchNorm2d(128)  # Batch Normalisation

        self.layer13 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Third convolution
        self.layer14 = nn.ReLU()  # Apply ReLU activation
        self.layer15 = nn.MaxPool2d(2, 2)  # Pooling after third convolution
        self.layer16 = nn.BatchNorm2d(256)  # Batch Normalisation
        self.layer17 = nn.Dropout(0.3)

        self.layer18 = nn.Linear(256 * 2 * 2, 256)
        self.layer19 = nn.ReLU()
        self.final_layer = nn.Linear(256, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)

        x = x.view(x.size(0), -1)  # flatten data
        x = self.layer18(x)
        x = self.layer19(x)

        x = self.final_layer(x)
        return x
