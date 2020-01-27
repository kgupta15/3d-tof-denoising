
% Given the scalar scene kernel matrix KScalar, the scene specification and 
% the modulation frequency f, this function computes the vector scene 
% kernel matrix KVector. The computations
% are performed by considering the scene piecewise - not the whole scene at
% once. This is done in order to avoid large memory requirements. The
% arguments PatchStart and PatchEnd are the starting and end indices of the
% receiving patches for which KScalar is computed. The computed matrix has
% a size (PatchEnd-PatchStart+1 X NumPatches)
% 
% The row dimension of KVector corresponds to the receiving patch. The
% column dimension corresponds to the sending patch. 


function KVector    = ComputeSceneKernelVectorFastPieceWise(KScalar, Scene, f, PatchStart, PatchEnd)

NumPatches      = numel(Scene.X);                   % total number of patches

% Computing patch-wise distances
XMat            = repmat(reshape(Scene.X(PatchStart:PatchEnd), PatchEnd-PatchStart+1, 1), [1 NumPatches]);
YMat            = repmat(reshape(Scene.Y(PatchStart:PatchEnd), PatchEnd-PatchStart+1, 1), [1 NumPatches]);
ZMat            = repmat(reshape(Scene.Z(PatchStart:PatchEnd), PatchEnd-PatchStart+1, 1), [1 NumPatches]);

XMatP           = repmat(Scene.X(:)', [PatchEnd-PatchStart+1, 1]);
YMatP           = repmat(Scene.Y(:)', [PatchEnd-PatchStart+1, 1]);
ZMatP           = repmat(Scene.Z(:)', [PatchEnd-PatchStart+1, 1]);

DistMat         = sqrt((XMat - XMatP).^2 + (YMat - YMatP).^2 + (ZMat - ZMatP).^2);

clear XMat YMat ZMat XMatP YMatP ZMatP

% Making the vector matrix
KVector         = KScalar .* exp(-1i*2*pi*DistMat*f/(3e8));

clear DistMat