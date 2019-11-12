import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class Model_A(nn.Module):
    def __init__(self, config):
        super(Model_A, self).__init__()
        self.config = config
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),            
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),            
            nn.MaxPool2d(kernel_size=2, stride=2))
        # self.fc = nn.Linear(in_features=7*7*32, out_features=self.config.data.num_classes)
        self.fc = nn.Linear(in_features=16*16*32, out_features=2)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class Model_B(nn.Module):
    def __init__(self, config):
        pass

    def forward(self, x):
        pass

config = []
x = torch.from_numpy(np.random.rand(1, 1, 64, 64))
model = Model_A(config)
model.double()
model.forward(x.double())