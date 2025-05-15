import torch
import torchvision
from torchvision import transforms

from model import IMLONetwork


def test_model(model):
    model.eval()  # evaluation mode
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32)
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = 100 * correct_predictions / total_predictions
    return accuracy





if __name__ == '__main__':
    model = IMLONetwork()
    model.load_state_dict(torch.load("IMLO_Coursework.pth"))
    print(f'Testing Accuracy: {test_model(model)}%')
