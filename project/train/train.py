import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cv2
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
from myDataset import MyDataset
from MyClass import Net
import argparse
import time

random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', help="#epoch")
parser.add_argument('--model', help="model name")
parser.add_argument('--RGB', help="rgb")
#parser.add_argument('--aug', default = True, help="augmentation")

args = parser.parse_args()

RGB = args.RGB
model = args.model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
date = str(time.localtime().tm_mon)+str(time.localtime().tm_mday)
rgb = "3channel" if eval(RGB) == True else "1channel"
description = date + "-" + model + "-" + rgb

# hyperparameters
batch_size = 1024
lr = 1e-2
epochs = int(args.epoch)

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize([0.7552, 0.7443, 0.7359], [0.3141, 0.3169, 0.3237]),
	])


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
	filename = './ckpt/{}-{}.pt'.format(desc, num)
	torch.save(model.state_dict(), filename)
	print("=> Saving checkpoint : {}".format(filename))


# fit
def fit(epochs, model, criterion, optimzier, train_loader, val_loader, description):
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
		accuracy_v = sum(val_accuracy) / len(val_accuracy); val_acc.append(accuracy_v)
		
		if epoch % 10 == 0:
			# checkpoint = {'state_dict' : model.state_dict()}
			save_checkpoint(description,model, epoch)
			print(f"[Epoch:{epoch}/{epochs}]\n[train] cost : {cost_t:<10.4f} accuracy : {accuracy_t:<10.4f}\n [dev]  cost : {cost_v:<10.4f} accuracy : {accuracy_v:<10.4f}\n")

	return train_cost, train_acc, val_cost, val_acc

trainset = datasets.ImageFolder(root = "dataset", transform = transform)
#print(trainset)
train_data, val_data = torch.utils.data.random_split(trainset, [20000, 4805])


train_loader = DataLoader(train_data, batch_size=batch_size, shuffle = True)
val_loader = DataLoader(val_data, batch_size = batch_size)


# define model
# print(model, RGB)
model = Net(model = model, RGB = eval(RGB)).to(device)

# optimizer
optimizer = optim.Adam(model.parameters())

# loss function
criterion = nn.CrossEntropyLoss()

train_cost, train_acc, val_cost, val_acc = fit(epochs, model, criterion, optimizer, train_loader, val_loader, description)

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
plt.savefig(f'Cost, Acc Graph : {description}.png')

# test
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((28, 28)),
	])

dataset = MyDataset(csv_file = 'testing.csv', root = 'targets', transform = transform)
test_loader = DataLoader(dataset, batch_size = batch_size)

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

with open("train_log.txt","a") as f:
	f.write(f"\nModel : {description}     test Accuracy : {test_accuracy:.4f}(Epoch : {epochs})\n")
print(f"\nModel : {description}     test Accuracy : {test_accuracy:.4f}\n")
