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

"""

Model A
=======

Input  : False Depth Map
Output : Parameter List (present in config file)

Model B
=======

Input  : Parameter List, False Depth Map
Output : True Depth Map

"""

def loadTrainingData(args):
	pass

def loadTestData(args):
	pass

class Flat_ModelA(Dataset):
	def __init__(self, args, train=True, transform=None):
		self.args = args
		if train == True:
			self.fdm, self.parameters = loadTrainingData(args)
		else:
			self.fdm, self.parameters = loadTestData(args)
		self.data_size = 0
		self.transform = transform

	def __getitem__(self, index):
		return (self.fdm, self.parameters) 

	def __len__(self):
		return self.data_size

class Flat_ModelB(Dataset):
	def __init__(self, args, train=True, transform=None):
		self.args = args
		if train == True:
			self.fdm, self.parameters, self.tdm = loadTrainingData(args)
		else:
			self.fdm, self.parameters, self.tdm = loadTestData(args)
		self.data_size = 0
		self.transform = transform

	def __getitem__(self, index):
		return (self.fdm, self.parameters, self.tdm)

	def __len__(self):
		return self.data_size