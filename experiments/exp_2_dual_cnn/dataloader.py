#!/usr/bin/env python

import os
from os import listdir
from os.path import isfile, join
import glob
import pickle
import imageio
import cv2
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

static = "/storage-pod/3d-tof-denoising/data/FLAT/FLAT/trans_render/static"
dynamic = "/storage-pod/3d-tof-denoising/data/FLAT/FLAT/trans_render/dyn"
full = "/storage-pod/3d-tof-denoising/data/FLAT/FLAT/kinect/full"
gt = "/storage-pod/3d-tof-denoising/data/FLAT/FLAT/kinect/gt" 

static_data = []
dynamic_data = []

"""
os.chdir(static)
for f in glob.glob("*.pickle"):
	with open(f,'rb') as file:
		data = pickle.load(file)
		static_data.append(data)

os.chdir(dyn)
for f in glob.glob("*.pickle"):
	with open(f,'rb') as file:
		data = pickle.load(file)
		dynamic_data.append(data)
"""

full_files = [f for f in listdir(full) if isfile(join(full, f))]
gt_files   = [f for f in listdir(gt) if isfile(join(gt, f))]
image_files = list(set(full_files) & set(gt_files))

def loadTrainingData_A(args):
	fdm = []
	parameters = []
	for i in image_files:
		try:
			with open(join(static, i + ".pickle"),'rb') as file:
				data = pickle.load(file)
				param = {}
				param['scene'] = data['scene']
				param['prop_idx'] = data['prop_idx']
				param['cam'] = data['cam']
				parameters.append(param)
			
				false_dm = np.fromfile(join(full, i), dtype=np.int32)
				false_dm = false_dm.reshape((424, 512, 9)).astype(np.float32)

		except:
			print('[!] File {} not found'.format(i))

	return (fdm, parameters)

def loadTestData_A(args):
	fdm = []
	parameters = []
	for i in image_files:
		try:
			with open(join(static, i + ".pickle"),'rb') as file:
				data = pickle.load(file)
				param = {}
				param['scene'] = data['scene']
				param['prop_idx'] = data['prop_idx']
				param['cam'] = data['cam']
				parameters.append(param)

			false_dm = np.fromfile(join(full, i), dtype=np.int32)
			false_dm = false_dm.reshape((424, 512, 9)).astype(np.float32)
	
		except:
			print('[!] File {} not found'.format(i))

	return (fdm, parameters)

def loadTrainingData_B(args):
	fdm = []
	tdm = []
	parameters = []
	for data in static_data:
		fdm.append(data['depth_true'])
		tdm.append(data['depth_true'])
		param = {}
		param['scene'] = data['scene']
		param['prop_idx'] = data['prop_idx']
		param['cam'] = data['cam']
		parameters.append(param)
	return (fdm, parameters, tdm)


def loadTestData_B(args):
	fdm = []
	tdm = []
	parameters = []
	for data in dynamic_data:
		fdm.append(data['depth_true'])
		tdm.append(data['depth_true'])
		param = {}
		param['scene'] = data['scene']
		param['prop_idx'] = data['prop_idx']
		param['cam'] = data['cam']
		parameters.append(param)
	return (fdm, parameters, tdm)

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

class Flat_ModelA(Dataset):
	def __init__(self, args, train=True, transform=None):
		self.args = args
		if train == True:
			self.fdm, self.parameters = loadTrainingData_A(args)
		else:
			self.fdm, self.parameters = loadTestData_A(args)
		self.data_size = 0
		self.transform = transform

	def __getitem__(self, index):
		return (self.fdm[index], self.parameters[index]) 

	def __len__(self):
		return self.data_size

class Flat_ModelB(Dataset):
	def __init__(self, args, train=True, transform=None):
		self.args = args
		if train == True:
			self.fdm, self.parameters, self.tdm = loadTrainingData_B(args)
		else:
			self.fdm, self.parameters, self.tdm = loadTestData_B(args)
		self.data_size = len(self.fdm)
		self.transform = transform

	def __getitem__(self, index):
		return (self.fdm[index], self.parameters[index], self.tdm[index])

	def __len__(self):
		return self.data_size