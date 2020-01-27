
% Given a scene specification, this function computes the scalar scene
% kernel matrix KScalar. For details, see the paper "Shape from
% interreflections". The computations
% are performed by considering the scene piecewise - not the whole scene at
% once. This is done in order to avoid large memory requirements. The
% arguments PatchStart and PatchEnd are the starting and end indices of the
% receiving patches for which KScalar is computed. The computed matrix has
% a size (PatchEnd-PatchStart+1 X NumPatches)
%
% The row dimension of KScalar corresponds to the receiving patch. The
% column dimension corresponds to the sending patch. 


function KScalar    = ComputeSceneKernelScalarPieceWise(Scene, PatchStart, PatchEnd)

NumPatches      = numel(Scene.X);                   % total number of patches

% Computing patch-wise distances
XMat            = repmat(reshape(Scene.X(PatchStart:PatchEnd), PatchEnd-PatchStart+1, 1), [1 NumPatches]);
YMat            = repmat(reshape(Scene.Y(PatchStart:PatchEnd), PatchEnd-PatchStart+1, 1), [1 NumPatches]);
ZMat            = repmat(reshape(Scene.Z(PatchStart:PatchEnd), PatchEnd-PatchStart+1, 1), [1 NumPatches]);

XMatP           = repmat(Scene.X(:)', [PatchEnd-PatchStart+1, 1]);
YMatP           = repmat(Scene.Y(:)', [PatchEnd-PatchStart+1, 1]);
ZMatP           = repmat(Scene.Z(:)', [PatchEnd-PatchStart+1, 1]);

DistMat         = sqrt((XMat - XMatP).^2 + (YMat - YMatP).^2 + (ZMat - ZMatP).^2);

% Computing angle between line joining two patches and the normal of the receiving patch
NXMat           = repmat(reshape(Scene.NX(PatchStart:PatchEnd), PatchEnd-PatchStart+1, 1), [1 NumPatches]);
NYMat           = repmat(reshape(Scene.NY(PatchStart:PatchEnd), PatchEnd-PatchStart+1, 1), [1 NumPatches]);
NZMat           = repmat(reshape(Scene.NZ(PatchStart:PatchEnd), PatchEnd-PatchStart+1, 1), [1 NumPatches]);
cosThetaOneMat  = ((XMatP-XMat) .* NXMat + (YMatP-YMat) .* NYMat + (ZMatP-ZMat) .* NZMat)  ./ (DistMat+eps);
cosThetaOneMat(cosThetaOneMat<0) = 0;
cosThetaOneMat(cosThetaOneMat>1) = 1;

clear NXMat NYMat NZMat

% Computing angle between line joining two patches and the normal of the sending patch
NXMatP          = repmat(Scene.NX(:)', [PatchEnd-PatchStart+1, 1]);
NYMatP          = repmat(Scene.NY(:)', [PatchEnd-PatchStart+1, 1]);
NZMatP          = repmat(Scene.NZ(:)', [PatchEnd-PatchStart+1, 1]);

cosThetaTwoMat  = ((XMat-XMatP) .* NXMatP + (YMat-YMatP) .* NYMatP + (ZMat-ZMatP) .* NZMatP)  ./ (DistMat+eps);
cosThetaTwoMat(cosThetaTwoMat<0) = 0;
cosThetaTwoMat(cosThetaTwoMat>1) = 1;

clear XMat YMat ZMat XMatP YMatP ZMatP NXMatP NYMatP NZMatP

% Make the matrix of areas of the sending patches
AreaMat         = repmat(Scene.Areas(:)', [PatchEnd-PatchStart+1 1]);       % This is because we need to multiply by areas of the sending patches

% Computing the final kernel matrix
KScalar         = (cosThetaOneMat .* cosThetaTwoMat .* AreaMat) ./ (DistMat.^2+eps);

clear costThetaOneMat cosThetaTwoMat AreaMat DistMat