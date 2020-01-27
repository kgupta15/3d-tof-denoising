
% This script performs direct global estimation from time-of-flight images
clear; clc

% Directory name containing the images
dirname     = ['./Images/Set-001-Vector'];
imPrefix    = 'Image-Total-Phase-';                     % Assuming the naming convention used in file SimulateImageScript.m
indices     = [0, 90, 180, 270];
indexLength = 3;
imSuffix    = '.tiff';
nr          = 2000;
nc          = 2000;
numChannels = 3;
phases      = [0:3]*pi/2;
outdirname  = dirname;
IScaleD     = 0.35;                                     % Scale factor applied to both the direct and global images for avoiding saturation
IScaleG     = 0.35;
medFiltParam= [1 1];

[IDirect, IGlobal]     = DirectGlobalEstimationFunc(dirname, imPrefix, indices, indexLength, imSuffix, nr, nc, numChannels, phases, outdirname, IScaleD, IScaleG, medFiltParam);