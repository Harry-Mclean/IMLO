import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn


#Variables to configure
epoch_count = 2

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


if __name__ == '__main__':
    # Initialises Model
    model = IMLONetwork()

    # sets transforms for images
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Downloads dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Loads dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    loss_function = nn.CrossEntropyLoss()  # Loss function uses cross entropy loss
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epoch_count):
        training_loss = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimiser.step()

            training_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {training_loss / 100:.3f}")
                training_loss = 0.0
