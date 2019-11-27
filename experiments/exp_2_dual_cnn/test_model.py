#!/usr/bin/env python

import os
import yaml
from PIL import Image
import numpy as np
import torch
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