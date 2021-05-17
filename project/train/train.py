import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import os
from torchvision.datasets import MNIST

from MyClass import Net

############################# init



random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', help="#epoch")
parser.add_argument('--model', help="model name")
parser.add_argument('--RGB', help="rgb")
parser.add_argument('--memo', default = '', help="note")


args = parser.parse_args()

RGB = args.RGB
model = args.model
memo = args.memo

device = 'cuda' if torch.cuda.is_available() else 'cpu'
date = str(time.localtime().tm_mon)+str(time.localtime().tm_mday)
rgb = "3channel" if eval(RGB) == True else "1channel"
description = date + "-" + model + "-" + rgb
if memo != "":
	description = date + "-" + model + "-" + rgb + "-" + memo
batch_size = 2 ** 13
lr = 1e-2
epochs = int(args.epoch)



############################# utils

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
def save_checkpoint(desc, model, num):
	path = f"./ckpt/{desc}"
	if not os.path.exists(path):
            os.makedirs(path)
	filename = path + f'/{num}.pt'
	torch.save(model.state_dict(), filename)
	print("=> Saving checkpoint : {}".format(filename))


# fit
def fit(epochs, model, criterion, optimzier, scheduler, train_loader, val_loader, description):
	train_cost = []
	train_acc = []
	val_cost = []
	val_acc = []
	print("=============Start Training=============")
	for epoch in range(epochs+1):
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
		scheduler.step(cost_v)
		accuracy_v = sum(val_accuracy) / len(val_accuracy); val_acc.append(accuracy_v)
		
		if epoch % 10 == 0:
			save_checkpoint(description,model, epoch)
			print(f"[Epoch:{epoch}/{epochs}]\n[train] cost : {cost_t:<10.4f} accuracy : {accuracy_t:<10.4f}\n [dev]  cost : {cost_v:<10.4f} accuracy : {accuracy_v:<10.4f}\n")
		

	return train_cost, train_acc, val_cost, val_acc



############################# Train

transform_custom = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize([0.7039, 0.6939, 0.6866], [0.3128, 0.3136, 0.3185]),
	])
transform_mnist = transforms.Compose([
	transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081]),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomRotation(10),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))]),
    transforms.RandomApply([transforms.RandomErasing()]),
	])

# load train dataset
trainset_custom = datasets.ImageFolder(root = "dataset", transform = transform_custom)
trainset_mnist = MNIST(root = './', train=True, download=True, transform = transform_mnist)
trainset_mnist, _ = torch.utils.data.random_split(trainset_mnist, [20000, len(trainset_mnist) - 20000])
trainset = torch.utils.data.ConcatDataset([trainset_mnist, trainset_custom])

train_data, val_data = torch.utils.data.random_split(trainset, [len(trainset)-10000,10000])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle = True)
val_loader = DataLoader(val_data, batch_size = batch_size)


# define model
model = Net(model = model, RGB = eval(RGB)).to(device)
#model.load_state_dict(torch.load("ckpt/517-3M-1channel/50.pt"))

# optimizer
optimizer = optim.Adam(model.parameters(), lr = lr)

# lr scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor= 0.1, patience=5, verbose=True)

# loss function
criterion = nn.CrossEntropyLoss()

# train
train_cost, train_acc, val_cost, val_acc = fit(epochs, model, criterion, optimizer, scheduler, train_loader, val_loader, description)

# save graph
plt.figure(figsize = (12,12))
plt.suptitle(description)
plt.subplot(211)
plt.plot(np.arange(len(train_cost)), train_cost, color = 'r', label = "train")
plt.plot(np.arange(len(train_cost)), val_cost, color = 'b', label = "val")
plt.legend(); plt.title(f"Cost Graph"); plt.xlabel("epoch"); plt.ylabel("cost")

plt.subplot(212)
plt.plot(np.arange(len(train_acc)), train_acc, color = 'r', label = "train")
plt.plot(np.arange(len(train_acc)), val_acc, color = 'b', label = "val")
plt.legend(); plt.title(f"Acc Graph"); plt.xlabel("epoch"); plt.ylabel("acc")
plt.savefig(f'./graphs/Cost, Acc Graph : {description}.png')



############################# Test

transform_custom = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((28, 28)),
	])
transform_mnist = transforms.Compose([
	transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Resize((28, 28)),
	])

# load test dataset
testset_custom = datasets.ImageFolder(root = "targets", transform = transform_custom)
testset_mnist = MNIST(root = './', train=False, download=True, transform = transform_mnist)
testset_mnist, _ = torch.utils.data.random_split(testset_mnist, [200, len(testset_mnist) - 200])

testset = torch.utils.data.ConcatDataset([testset_mnist, testset_custom])

test_loader = DataLoader(testset, batch_size=batch_size)

# test
ckpt_path = f"ckpt/{description}/"
result = {}
for pt in os.listdir(ckpt_path):
	model.load_state_dict(torch.load(ckpt_path + pt))	
	model.eval()
	acc_list = []
	with torch.no_grad():
		for data, target in test_loader:
			data = data.to(device)
			target = target.to(device)
			
			pred = model(data)

			acc = get_acc(pred, target)
			acc_list.append(acc)
	test_accuracy = sum(acc_list) / len(acc_list)
	result[pt] = test_accuracy
result = sorted(result.items(), reverse = True, key = lambda item : item[1])
best = result[0]

# save log
with open("train_log.txt","a") as f:
	f.write(f"\nModel : {description}     test Accuracy : {best[1]:.4f} (epoch : {best[0][:-3]})\n")
print(f"\nModel : {description}     test Accuracy : {best[1]:.4f}\n")
