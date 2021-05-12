import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2
import albumentations as A
import numpy as np
from PIL import Image
from torchvision.datasets import MNIST

from MyClass import Net
from utils import get_acc, loss_batch, fit

# hyperparameters
batch_size = 128
lr = 1e-3
epochs = 1000

# make data have 3 channels
transform = transforms.Compose([
	transforms.Grayscale(num_output_channels = 3),
	transforms.ToTensor(),
	])

train_data = MNIST(root = "./", train = True, transform = transform, download = True)
test_data = MNIST(root = "./", train = False, transform = transform, download = True)

train_data, val_data = torch.utils.data.random_split(train_data, [50000, 10000])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle = True)
val_loader = DataLoader(val_data, batch_size = batch_size)
test_loader = DataLoader(test_data, batch_size = batch_size)

# define model
model = Net().to(device)

# optimizer
optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9)

# loss function
criterion = nn.CrossEntropyLoss()

train_cost, train_acc, val_cost, val_acc = fit(epochs, model, criterion, optimizer, train_loader, val_loader)