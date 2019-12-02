#!/usr/bin/env python

import os
from os.path import join
import yaml
from PIL import Image
import numpy as np
import torch
from torchvision import transforms, utils
from mapper import *
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('config_a.yaml', 'r') as file:
    stream = file.read()
    config_dict = yaml.safe_load(stream)
    config_a = mapper(**config_dict)
with open('config_b.yaml', 'r') as file:
    stream = file.read()
    config_dict = yaml.safe_load(stream)
    config_b = mapper(**config_dict)    

"""
model_a = Model_A(config_a)
model_a.to(device)
# os.chdir('../../data')
# img = np.array(Image.open('room_tdm.jpg'))
img = np.random.random((424, 512))
img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis=1)
params = model_a(torch.from_numpy(img).type(torch.FloatTensor).to(device))
print(params)
"""

"""
model_b = Model_B(config_b)
model_b.to(device)
os.chdir('../../data')
# img = np.array(Image.open('room_tdm.jpg'))
img = np.random.rand(424, 512)
img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis=1)
out_img = model_b(torch.from_numpy(img).type(torch.FloatTensor).to(device), torch.from_numpy(np.random.random((1,1,8,128))).type(torch.FloatTensor).to(device))
print(out_img.shape)
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model_B(config_b)
model = nn.DataParallel(model).to(device)
os.chdir('/storage-pod/3d-tof-denoising/experiments/exp_2.1_dual_cnn/runs_b/checkpoints')
bestckpt_file = 'best_checkpoint.pth.tar'
ckpt_file = 'checkpoint.pth.tar'
ckpt = torch.load(bestckpt_file)
model.load_state_dict(ckpt['state_dict'])
os.chdir('/storage-pod/3d-tof-denoising/data/FLAT/FLAT/kinect/reflection')
fdm = np.fromfile('./1520215621736999', dtype=np.int32)
fdm = Image.fromarray(fdm.reshape((424, 512, 9))[:,:,1])
os.chdir('/storage-pod/3d-tof-denoising/data/FLAT/FLAT/kinect')
params = np.load('data.npy')
param = np.array(params[0, 1:])
param = np.where(param == '-point-light-source', 1, param).astype(np.float64)
transform = transforms.Compose([transforms.ToTensor()]) 
out = model(transform(fdm).view(1, 1, 424, 512).type(torch.FloatTensor).to(device), torch.from_numpy(param).type(torch.FloatTensor).to(device))
out = out.cpu().detach().numpy()
out = np.reshape(out, (424, 512))
print(out.shape)
plt.imsave('output.jpg', out)