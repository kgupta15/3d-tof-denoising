
% This function performs direct global estimation from time-of-flight
% images. 
%
% Inputs: 
% dirname - input directory, 
% imPrefix - prefix of input images
% indices - indices of the input images
% indexLength - length of indices of input images
% imSuffix - suffix of input images
% nr, nc - size of input images (rows and columns)
% numChannels - number of color channels of input images
% phases - phases at which measurements are taken
% outdirname - name of output directory where result direct and global images are stored
% IScaleD - scale factor applied to final direct images before saving
% IScaleG - scale factor applied to final global images before saving
% medFiltParam - parameters for median filtering the image before saving

function [IDirect, IGlobal]     = DirectGlobalEstimationFunc(dirname, imPrefix, indices, indexLength, imSuffix, nr, nc, numChannels, phases, outdirname, IScaleD, IScaleG, medFiltParam)

mkdir(outdirname);

% Read images
IMat        = zeros(nr, nc, numChannels, numel(phases));
for i=1:numel(indices)
    imname          = [dirname, '\', imPrefix, sprintf(['%0', num2str(indexLength), 'd'], indices(i)), imSuffix];
    IMat(:,:,:,i)   = im2double(imread(imname));
end

% Making the measurement matrix
A           = [ones(numel(phases), 1),  cos(phases'),  -sin(phases')];

% Computing sinusoid parameters, and then the direct global components
IDirect     = zeros(nr, nc, numChannels);
IGlobal     = zeros(nr, nc, numChannels);

for c=1:numChannels
    I               = squeeze(IMat(:,:,c,:));
    I               = reshape(I, [nr*nc, numel(phases)])';
    
    Params          = A\I;
    
    Offsets         = Params(1,:);
    Amps            = sqrt(Params(2,:).^2 + Params(3,:).^2);

    IDirect(:,:,c)  = medfilt2(reshape(Amps, [nr, nc]) * 8, medFiltParam);
    IGlobal(:,:,c)  = medfilt2(reshape(Offsets, [nr, nc]) * 4 - IDirect(:,:,c), medFiltParam);
end

% Writing images
imwrite(im2uint16(IDirect * IScaleD), [outdirname, '\IDirect.tiff']);
imwrite(im2uint16(IGlobal * IScaleG), [outdirname, '\IGlobal.tiff']);