#!/usr/bin/env python

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class FlatDataset(Dataset):
	def __init__(self, args, train=True, transform=None):
		self.args = args
		if train == True:
			pass
		else:
			pass
		self.data_size = 0
		self.transform = transform

	def __getitem__(self, index):
		return 0

	def __len__(self):
		return self.data_size