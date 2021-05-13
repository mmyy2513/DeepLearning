import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import argparse
from myDataset import MyDataset
from MyClass import Net

parser = argparse.ArgumentParser(description='Load Weight and Evaluate')
parser.add_argument('--ckpt', help="Path of Checkpoint")
parser.add_argument('--model', help="model name")
parser.add_argument('--RGB', help="rgb")

args = parser.parse_args()

RGB = args.RGB
model = args.model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128

# acc function
def get_acc(pred, label):
	preds = torch.argmax(pred, dim = 1)
	return (label == preds).float().mean()

def load_model(ckpt):
	model = Net(model = args.model, RGB = eval(RGB))
	model.load_state_dict(torch.load(ckpt))
	return model

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((28, 28)),
	])

dataset = MyDataset(csv_file = 'testing.csv', root = 'targets', transform = transform)
test_loader = DataLoader(dataset, batch_size = batch_size)


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
		# import matplotlib.pyplot as plt 
		# plt.imshow(data.cpu().reshape(3,28,28).numpy().transpose(1,2,0))
		# plt.show()
		# print(pred.argmax(1).item()," ----- ", target.item())
test_accuracy = sum(acc_list) / len(acc_list)
print(f"\nTest Accuracy : {test_accuracy:.4f}\t\t(checkpoint = {args.ckpt})")
