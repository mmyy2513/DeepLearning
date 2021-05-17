import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from torchvision.datasets import MNIST

transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = datasets.ImageFolder(root = "dataset", transform = transform)
#trainset = MNIST(root = './', train=True, download=True, transform = transform)
train_loader = DataLoader(trainset)
def get_mean_std(loader):
    # VAR[X] = E[X**2] - E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0,0, 0
    
    for data, _ in loader:
        channels_sum += torch.mean(data, dim = [0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim = [0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    
    return mean, std

mean, std = get_mean_std(train_loader)
print(mean, std)