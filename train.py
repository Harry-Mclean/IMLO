import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
from test import test_model
import time
from model import IMLONetwork

#Variables to configure
epoch_count = 50

if __name__ == '__main__':
    device = torch.device('cpu') # sets device as CPU


    # Initialises Model
    model = IMLONetwork().to(device)

    # sets transforms for images
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Downloads dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Loads dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    loss_function = nn.CrossEntropyLoss()  # Loss function uses cross entropy loss
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    for epoch in range(epoch_count):
        training_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # make sure inputs and labels are on cpu
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
    time_taken = time.time() - start_time
    hours = time_taken // 3600
    minutes = (time_taken % 3600) // 60
    seconds = time_taken % 60

    print(f"Training complete in {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")

    print('Finished Training')
    print(f'Testing Accuracy: {test_model(model)}%')
