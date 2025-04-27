import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
from test import test_model

from model import IMLONetwork
#Variables to configure
epoch_count = 2



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
        correct_predictions = 0
        total_predictions = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimiser.step()

            training_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability (predicted class)
            correct_predictions += (predicted == labels).sum().item()  # Compare predicted with ground truth labels
            total_predictions += labels.size(0)  # Add the batch size to the total count

            if i % 100 == 99:  # print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {training_loss / 100:.3f}")
                print(f"Training Accuracy: {correct_predictions / total_predictions * 100:.2f}%")
                training_loss = 0.0

    torch.save(model.state_dict(), 'IMLO_Coursework.pth')
    print('Finished Training')
    print(f'Testing Accuracy: {test_model(model)}%')
