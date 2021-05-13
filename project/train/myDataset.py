import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

class MyDataset(Dataset):
	def __init__(self, csv_file, root, transform = None):
		self.annot = pd.read_csv(csv_file)
		self.root = root
		self.transform = transform

	def __len__(self):
		return len(self.annot)

	def __getitem__(self, idx):
		img_path = os.path.join(self.root, self.annot.iloc[idx, 0])
		img = io.imread(img_path)
		label = torch.tensor(int(self.annot.iloc[idx, 1]))

		if self.transform:
			img = self.transform(img)

		return (img, label)