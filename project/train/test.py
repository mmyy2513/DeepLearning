import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import argparse

from MyClass import Net
# from utils import get_acc, loss_batch, fit
parser = argparse.ArgumentParser(description='Load Weight and Evaluate')
parser.add_argument('--ckpt', help="Path of Checkpoint")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128

# acc function
def get_acc(pred, label):
	preds = torch.argmax(pred, dim = 1)
	return (label == preds).float().mean()

def load_model(ckpt):
	model = Net()
	model.load_state_dict(torch.load(ckpt))
	return model

transform = transforms.Compose([
	transforms.Grayscale(num_output_channels = 3),
	transforms.ToTensor(),
	])

test_data = MNIST(root = "./", train = False, transform = transform, download = True)
test_loader = DataLoader(test_data, batch_size = batch_size)

model = load_model(args.ckpt)
model = model.to(device)
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
print(f"\nTest Accuracy : {test_accuracy:.4f}\t\t(checkpoint = {args.ckpt})")
