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

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.in_dim = 28 * 28 * 1
		self.out_dim = 10

		self.linear = nn.Linear(self.in_dim, self.out_dim)

	def forward(self, x):
		x = transforms.Grayscale(num_output_channels = 1)(x)
		
		x = x.view(x.shape[0], -1)
		x = self.linear(x)
		return x


		