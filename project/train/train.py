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
# from utils import get_acc, loss_batch, fit

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# acc function
def get_acc(pred, label):
	preds = torch.argmax(pred, dim = 1)
	return (label == preds).float().mean()

# get loss, acc of one batch
def loss_batch(model, criterion, data, target, optimizer = None):
	data = data.to(device)
	target = target.to(device)

	pred = model(data)
	loss = criterion(pred, target) 
	acc = get_acc(pred, target)

	if optimizer is not None:
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	return loss.item(), acc


# save ckpt
def save_checkpoint(desc, state, num):
	filename = './ckpt/{}-{}.pth.tar'.format(desc,num)
	torch.save(state, filename)
	print("=> Saving checkpoint : {}".format(filename))


# fit
def fit(epochs, model, criterion, optimzier, train_loader, val_loader, description):
	train_cost = []
	train_acc = []
	val_cost = []
	val_acc = []

	for epoch in range(epochs):
		# train
		model.train()
		train_loss = []
		train_accuracy = []

		for data, target in train_loader:
			loss, acc = loss_batch(model, criterion, data, target, optimizer)
			train_loss.append(loss); train_accuracy.append(acc)

		cost_t = sum(train_loss) / len(train_loss); train_cost.append(cost_t)
		accuracy_t = sum(train_accuracy) / len(train_accuracy); train_acc.append(accuracy_t)

		# valid
		model.eval()
		val_loss = []
		val_accuracy = []
		with torch.no_grad():
			for data, target in val_loader:
				loss, acc = loss_batch(model, criterion, data, target)
				val_loss.append(loss); val_accuracy.append(acc)
		
		cost_v = sum(val_loss) / len(val_loss); val_cost.append(cost_v)
		accuracy_v = sum(val_accuracy) / len(val_accuracy); val_acc.append(accuracy_v)
		
		if epoch % 10 == 0:
			checkpoint = {'state_dict' : model.state_dict()}
			save_checkpoint(description,checkpoint, epoch)
			print(f"[Epoch:{epoch}/{epochs}]\n[train] cost : {cost_t:<10.4f} accuracy : {accuracy_t:<10.4f}\n [dev]  cost : {cost_v:<10.4f} accuracy : {accuracy_v:<10.4f}\n")

	return train_cost, train_acc, val_cost, val_acc

# hyperparameters
batch_size = 128
lr = 1e-1
epochs = 100

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
optimizer = optim.Adam(model.parameters())

# loss function
criterion = nn.CrossEntropyLoss()

train_cost, train_acc, val_cost, val_acc = fit(epochs, model, criterion, optimizer, train_loader, val_loader, "0512")