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


VGG_types = {
	'SIMPLE' : [32, 16, 0],
	'2M' : [64, 128, 'M', 256, 512, 'M', 2],
}

class Net(nn.Module):
	def __init__(self,  model, num_classes=10, init_weights=True, RGB = True):
		super(Net, self).__init__()
		n_pool = VGG_types[model].pop()
		last = VGG_types[model][-1] if type(VGG_types[model][-1]) == int else VGG_types[model][-2]
		self.RGB = RGB
		self.in_channels = 3 if RGB == True else 1
		self.num_classes = num_classes
		self.conv_layers = self.create_conv_layers(VGG_types[model])
		
		self.fc_layers = nn.Sequential(	
			nn.Linear(last * (28//(2**n_pool)) * (28//(2**n_pool)), 1000),
			nn.ReLU(),
			nn.Dropout(p = 0.5),
			nn.Linear(1000, num_classes)
			)
		if init_weights == True:
			self._initialize_weights()

	def forward(self, x):
		# print(x.shape)
		if self.RGB == False:
			x = transforms.Grayscale(num_output_channels = 1)(x)
		# print(x.shape)
		x = self.conv_layers(x)
		# print(x.shape)
		x = x.reshape(x.shape[0], -1)
		# print(x.shape)
		x = self.fc_layers(x)
		# print(x.shape)
		return x

	def _initialize_weights(self):
			# modules -> Sequential 모든 layer
			for m in self.modules():
				if isinstance(m, nn.Conv2d):
					# he_initialization
					nn.init.kaiming_normal_(m.weight,
											mode='fan_out',
											nonlinearity='relu')
					if m.bias is not None:
						nn.init.constant_(m.bias, 0)
				elif isinstance(m, nn.BatchNorm2d):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
				elif isinstance(m, nn.Linear):
					nn.init.normal_(m.weight, 0, 0.01)
					nn.init.normal_(m.bias, 0)

	def create_conv_layers(self, architecture):
		# print(architecture)
		layers = []
		in_channels = self.in_channels

		for x in architecture:
			if type(x) == int:  # conv layer
				out_channels = x

				layers += [
					nn.Conv2d(in_channels,
							  out_channels,
							  kernel_size=(3, 3),
							  stride=(1, 1),
							  padding=(1, 1)),
					nn.BatchNorm2d(x),
					nn.ReLU()
				]
				# channel num --> next input
				in_channels = x

			elif x == 'M':  # maxpooling
				layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

		return nn.Sequential(*layers)