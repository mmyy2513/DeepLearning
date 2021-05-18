import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
from MyClass import Net
from torchvision.datasets import MNIST
import numpy as np

random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description='Load Weight and Evaluate')
parser.add_argument('--ckpt', help="Path of Checkpoint")
parser.add_argument('--model', help="model name")
parser.add_argument('--RGB', help="rgb")

args = parser.parse_args()

RGB = False if args.RGB == "False" else True
#print(RGB)
model = args.model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1024

# acc function
def get_acc(pred, label):
	preds = torch.argmax(pred, dim = 1)
	return (label == preds).float().mean()

def load_model(ckpt, model, RGB):
	model = Net(model = model, RGB = RGB)
	model.load_state_dict(torch.load(ckpt))
	return model

# test
transform_custom = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((28, 28)),
	])
transform_mnist = transforms.Compose([
	transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Resize((28, 28)),
	])


testset_custom = datasets.ImageFolder(root = "targets", transform = transform_custom)
testset_mnist = MNIST(root = './', train=False, download=True, transform = transform_mnist)
testset_mnist, _ = torch.utils.data.random_split(testset_mnist, [200, len(testset_mnist) - 200])

testset = torch.utils.data.ConcatDataset([testset_mnist, testset_custom])

test_loader = DataLoader(testset, batch_size=batch_size, shuffle = True)

model = load_model(args.ckpt, args.model, RGB = RGB)
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
		#import matplotlib.pyplot as plt 
		#print(data.shape)
		# print(target.shape)
		# print(pred.shape)
		# for i in range(data.shape[0]):
		# 	plt.imshow(data[i].cpu().reshape(3,28,28).numpy().transpose(1,2,0))
		# 	plt.show()
		# 	print(pred.argmax(1)[i].item()," ----- ", target[i].item())
test_accuracy = sum(acc_list) / len(acc_list)
print(f"\nTest Accuracy : {test_accuracy:.4f}\t\t(checkpoint = {args.ckpt})")
