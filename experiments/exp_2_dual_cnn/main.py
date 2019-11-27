#!/usr/bin/env python

import sys
import os.path
import argparse
from argparse import RawTextHelpFormatter
from inspect import getsourcefile

import numpy as np
import yaml
import torch

import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
from torch.backends import cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from dataloader import Flat_ModelA, Flat_ModelB
from model import Model_A, Model_B
from trainer import Trainer
from evaluator import Evaluator
from mapper import *

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
parent_dir = parent_dir[:parent_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
from utils import *
sys.path.pop(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    np.random.seed(0)
    torch.manual_seed(0)

    with open('config_a.yaml', 'r') as file:
    	stream = file.read()
    	config_dict = yaml.safe_load(stream)
    	config_a = mapper(**config_dict)

    model = Model_A(config_a)
    plt.ion()

    if config_a.distributed:
    	model.to(device)
    	model = nn.parallel.DistributedDataParallel(model)
    elif config_a.gpu:
    	model = nn.DataParallel(model).to(device)
    else: return

    # Data Loading
    """
    train_dataset = torchvision.datasets.MNIST(root=os.path.join(parent_dir, 'data'),
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

    test_dataset = torchvision.datasets.MNIST(root=os.path.join(parent_dir, 'data'),
                                              train=False,
                                              transform=transforms.ToTensor())
    


    if config_a.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config_a.data.batch_size, shuffle=config_a.data.shuffle,
        num_workers=config_a.data.workers, pin_memory=config_a.data.pin_memory, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config_a.data.batch_size, shuffle=config_a.data.shuffle,
        num_workers=config_a.data.workers, pin_memory=config_a.data.pin_memory)
    """

    train_dataset = Flat_ModelA(args=config_a.data,
                                train=True,
                                transform=transforms.ToTensor())

    test_dataset  = Flat_ModelB(args=config_a.data,
                                train=False,
                                transform=transforms.ToTensor())

    if config_a.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config_a.data.batch_size, shuffle=config_a.data.shuffle,
        num_workers=config_a.data.workers, pin_memory=config_a.data.pin_memory, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config_a.data.batch_size, shuffle=config_a.data.shuffle,
        num_workers=config_a.data.workers, pin_memory=config_a.data.pin_memory)

    if args.train:
    	# trainer settings
    	trainer = Trainer(config_a.train, train_loader, model)
    	criterion = nn.CrossEntropyLoss().to(device)
    	optimizer = torch.optim.Adam(model.parameters(), config_a.train.hyperparameters.lr)
    	trainer.setCriterion(criterion)
    	trainer.setOptimizer(optimizer)
    	# evaluator settings
    	evaluator = Evaluator(config_a.evaluate, val_loader, model)
    	optimizer = torch.optim.Adam(model.parameters(), lr=config_a.evaluate.hyperparameters.lr, 
    		weight_decay=config_a.evaluate.hyperparameters.weight_decay)
    	evaluator.setCriterion(criterion)

    if args.test:
    	pass

    # Turn on benchmark if the input sizes don't vary
    # It is used to find best way to run models on your machine
    cudnn.benchmark = True
    start_epoch = 0
    best_precision = 0
    
    # optionally resume from a checkpoint
    if config_a.train.resume:
        [start_epoch, best_precision] = trainer.load_saved_checkpoint(checkpoint=None)

    # change value to test.hyperparameters on testing
    for epoch in range(start_epoch, config_a.train.hyperparameters.total_epochs):
        if config_a.distributed:
            train_sampler.set_epoch(epoch)

        if args.train:
            trainer.adjust_learning_rate(epoch)
            trainer.train(epoch)
            prec1 = evaluator.evaluate(epoch)

        if args.test:
        	pass

        # remember best prec@1 and save checkpoint
        if args.train:
            is_best = prec1 > best_precision
            best_precision = max(prec1, best_precision)
            trainer.save_checkpoint({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'best_precision': best_precision,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=None)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
	parser.add_argument('--train', type=str2bool, default='1', \
				help='Turns ON training; default=ON')
	parser.add_argument('--test', type=str2bool, default='0', \
				help='Turns ON testing; default=OFF')
	args = parser.parse_args()
	main(args)
