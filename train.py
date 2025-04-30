import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
from torch.utils.data import random_split

from test import test_model, training_test
import time
from model import IMLONetwork

# Variables to configure

epoch_count = 50
training_validation_split = 0.8

if __name__ == '__main__':
    device = torch.device('cpu')  # sets device as CPU

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

    # splits dataset into training set and validation set
    train_size = int(training_validation_split * len(trainset))
    val_size = len(trainset) - train_size
    train_data, val_data = random_split(trainset, [train_size, val_size])

    # Loads dataset
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

    loss_function = nn.CrossEntropyLoss()  # Loss function uses cross entropy loss
    optimiser = torch.optim.Adam(model.parameters(), lr=0.002)

    start_time = time.time()

    for epoch in range(epoch_count):
        model.train()
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
            training_accuracy = 100 * correct_predictions / total_predictions

        model.eval()  # Set model to evaluation mode
        val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():  # No gradients needed for validation
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

        validation_accuracy = 100 * correct_predictions / total_predictions
        print(f"[Epoch {epoch + 1}] Training Loss: {training_loss / len(trainloader):.3f}, "
              f"Training Accuracy: {training_accuracy:.2f}%, "
              f"Validation Accuracy: {validation_accuracy:.2f}%")

    torch.save(model.state_dict(), 'IMLO_Coursework.pth')
    time_taken = time.time() - start_time
    hours = time_taken // 3600
    minutes = (time_taken % 3600) // 60
    seconds = time_taken % 60

    print(f"Training complete in {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")

    print('Finished Training')
    print(f"Training Accuracy: {training_test(model)}%")
    print(f'Testing Accuracy: {test_model(model)}%')
