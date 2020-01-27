
clear all; clc
% This script performs time-of-flight phase shifting for depth computation.

% Give the full path of the directory containing the images
basedirname     = './Images';
outdirname      = [basedirname, '/Results/ShapeFiles'];             % output directory
mkdir(outdirname);
ShapeFName      = ['Results/ShapeFiles/Shape.mat'];                 % Name of the file containing the shape

freqIndices     = [1:2];                                            % Indices of frequencies used for reconstruction
shiftVec        = [0:3]*90;                                         % phase values for each frequency
nr              = 2000;                                             % camera resolution
nc              = 2000;
DepthRange      = [10:6500];                                        % Range of possible depth values (in millimeters)
subsampleFactor = 0.2;                                              % spatial subsampling factor during reconstruction. 
medfiltParams   = [1, 1];                                           % parameters for median-filtering the computed depth maps

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CameraAndLightSpecifications;                                       % Making the camera and light structure

freqVec         = Light.Freqs;
freqs           = freqVec(freqIndices);

dirPrefix       = 'Set-';
dirIndexLength  = 3;
dirSuffix       = '-Vector';

imPrefix        = 'Image-Total-Phase-';
imIndices       = shiftVec;
imIndexLength   = 3;
imSuffix        = '.tiff';

PhaseMaps       = zeros(nr*subsampleFactor, nc*subsampleFactor, numel(freqIndices));

for i=1:numel(freqIndices)
    dirname     = [basedirname, '\', dirPrefix, sprintf(['%0', num2str(dirIndexLength), 'd'], freqIndices(i)), dirSuffix];
    
    IMat        = zeros(nr*subsampleFactor, nc*subsampleFactor, numel(shiftVec));
    
    for j=1:numel(shiftVec)
        imname          = [dirname, '\', imPrefix, sprintf(['%0', num2str(imIndexLength), 'd'], shiftVec(j)), imSuffix];
        ITmp            = im2double(imread(imname));
        ITmp            = mean(ITmp, 3);
        IMat(:,:,j)     = imresize(ITmp, subsampleFactor, 'nearest');
    end
    
    PhaseMaps(:,:,i)  =  ComputePhaseMaps(IMat, shiftVec * pi/180);
    i
end

Depths      = PhaseMapsToDepths(freqs, PhaseMaps, DepthRange);
Depths      = medfilt2(Depths, medfiltParams);
Depths      = imresize(Depths, [nr nc]);

% Converting to shape from depths
Shape       = DepthsToShape(Cam, Depths);

% Saving shape file
save([basedirname, '\', ShapeFName], 'Shape')