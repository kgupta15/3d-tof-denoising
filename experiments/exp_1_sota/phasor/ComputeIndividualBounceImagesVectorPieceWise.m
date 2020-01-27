
% This function computes the complete (scalar+vector) individual bounce 
% images due to light originating from the light source. The computations
% are performed by considering the scene piecewise - not the whole scene at
% once. This is done in order to avoid large memory requirements. 
%
% Input: Structures Scene, Light, Cam. N is the number of bounces. Also, 
% FreqNum (which frequency) and CamPhase (phase of the reference signal - 
% assuming light source's phase is 0) (in degrees). outdir is the directory 
% containing the output files. 
% 
% NumPieces is the number of pieces in which to break the problem.
%
% Output: IIndividualBounce - tensor of size rxcx3xN, where rxc is the 
% image dimensions. 


function [IIndividualBounce]        = ComputeIndividualBounceImagesVectorPieceWise(Scene, Light, Cam, N, FreqNum, CamPhase, outdir, NumPieces)

mkdir(outdir);

%%%%%%%% Making First Bounce Scalar Component (due to constant offset of illumination sinusoid) of Radiance %%%%%%%%%%%%%%%%%
RadIndividualBounceScalar       = zeros(size(Scene.X, 1), size(Scene.X, 2), 3, N);

% Computing distances between light source and scene points
SceneLightDists   	= sqrt((Scene.X - Light.Pos(1)).^2 + (Scene.Y - Light.Pos(2)).^2 + (Scene.Z - Light.Pos(3)).^2);
SceneCamDists   	= sqrt((Scene.X - Cam.Pos(1)).^2 + (Scene.Y - Cam.Pos(2)).^2 + (Scene.Z - Cam.Pos(3)).^2);

% Computing dot product between scene normals and the line joining light source and scene point
CosTheta            = ((Light.Pos(1) - Scene.X) .* Scene.NX + (Light.Pos(2) - Scene.Y) .* Scene.NY + (Light.Pos(3) - Scene.Z) .* Scene.NZ) ./ SceneLightDists;

% Direct Scene Irradiances
IrradMat            = (Light.I0 * Light.Offsets(FreqNum)) .* CosTheta ./ SceneLightDists.^2;

% Direct Scene Radiances (first bounce)
for c=1:3
    RadIndividualBounceScalar(:,:,c,1)  = IrradMat .* Scene.Albedos(:,:,c) / pi;                                % Radiance due to direct illumination
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%% Making First Bounce Vector Component (due to oscillating component of illumination sinusoid) of Radiance %%%%%%%%%%%%%
RadIndividualBounceVector       = zeros(size(Scene.X, 1), size(Scene.X, 2), 3, N);

% Direct Scene Irradiances
IrradMat            = ((Light.I0 * Light.Amps(FreqNum)) .* CosTheta ./ SceneLightDists.^2) .* exp(-1i*2*pi*SceneLightDists*Light.Freqs(FreqNum)/3e8);

% Direct Scene Radiances (first bounce)
for c=1:3
    RadIndividualBounceVector(:,:,c,1)  = IrradMat .* Scene.Albedos(:,:,c) / pi;                                % Radiance due to direct illumination
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%% Making Subsequent Bounces (Both Scalar and Vector) %%%%%%%%%%%

for bounce=2:N
    for p=1:NumPieces
        
        NSceneRows          = size(Scene.X, 1);
        NSceneCols          = size(Scene.X, 2);
        NumColsPerPiece     = ceil(NSceneCols/NumPieces);       % Number of scene columns considered per piece
        
        StartCol            = (p-1)*NumColsPerPiece + 1;
        EndCol              = min(p*NumColsPerPiece, NSceneCols);
        
        PatchStart          = (StartCol-1)*NSceneRows + 1;      % Index of the starting patch
        PatchEnd            = (EndCol)*NSceneRows;              % Index of the ending patch
        
        KScalar             = ComputeSceneKernelScalarPieceWise(Scene, PatchStart, PatchEnd);
        KVector             = ComputeSceneKernelVectorFastPieceWise(KScalar, Scene, Light.Freqs(FreqNum), PatchStart, PatchEnd);
        
        for c=1:3
            
            % Making the Albedo Matrix
            Albedos       	= Scene.Albedos(:,StartCol:EndCol,c)/pi;
            AlbedoMat       = diag(Albedos(:));
            
            % scalar-component
            RadPrevScalar                                           = RadIndividualBounceScalar(:,:,c,bounce-1);                         	% Radiance due to previous bounce
            RadNewScalar                                            = AlbedoMat * KScalar * RadPrevScalar(:);                               % Making the new bounce
            RadIndividualBounceScalar(:,StartCol:EndCol,c,bounce)   = reshape(RadNewScalar, size(Scene.X, 1), EndCol-StartCol+1);           % reshaping the solution
            
            % vector-component
            RadPrevVector                                           = RadIndividualBounceVector(:,:,c,bounce-1);                         	% Radiance due to previous bounce
            RadNewVector                                            = AlbedoMat * KVector * RadPrevVector(:);                               % Making the new bounce
            RadIndividualBounceVector(:,StartCol:EndCol,c,bounce)   = reshape(RadNewVector, size(Scene.X, 1), EndCol-StartCol+1);           % reshaping the solution
            
            [bounce, p, c]
        end
        
        clear KScalar KVector
    end
end

% Add the phase delay to RadIndividualBounceVector due to propagation back to the sensor 
RadIndividualBounceVector       = RadIndividualBounceVector .* repmat(exp(-1i*2*pi*SceneCamDists*Light.Freqs(FreqNum)/3e8), [1 1 3 N]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%% Making the Images %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Multiplying scalar radiance by camera's modulation offset
RadIndividualBounceScalar       = RadIndividualBounceScalar * Cam.Offsets(FreqNum);

% Computing the phases of the vector radiance
RadiancePhases                  = zeros(size(Scene.X, 1), size(Scene.X, 2), 1, N);
for b=1:N
    for c=1:3
        RadiancePhasesTmp                                                   = acos(real(RadIndividualBounceVector(:,:,c,b) ./ (abs(RadIndividualBounceVector(:,:,c,b)) +eps)));
        RadiancePhasesTmp(imag(RadIndividualBounceVector(:,:,c,b))<0)       = 2*pi - RadiancePhasesTmp(imag(RadIndividualBounceVector(:,:,c,b))<0);
        RadiancePhases(:,:,c,b)                                            	= RadiancePhasesTmp;
    end
end

% Applying correlation to the vector radiance
RadIndividualBounceVector       = abs(RadIndividualBounceVector) * Cam.Amps(FreqNum) * 0.5 .* cos(RadiancePhases-CamPhase*pi/180);


% Computing image value
IIndividualBounce   = zeros(Cam.NRow, Cam.NCol, 3, N);
for b=1:N
    IIndividualBounce(:,:,:,b) = ConvertRadianceToImage(RadIndividualBounceScalar(:,:,:,b) + RadIndividualBounceVector(:,:,:,b), Scene, Cam);
%     imwrite(im2uint16(imresize(IIndividualBounce(:,:,:,b),1)), [outdir, '/Image-NBounce', sprintf('%02d', b), '-Phase-', sprintf('%03d', CamPhase), '.tiff'])
end

% IDirect     = ConvertRadianceToImage(RadIndividualBounceScalar(:,:,:,1) + RadIndividualBounceVector(:,:,:,1), Scene, Cam);
% IGlobal     = ConvertRadianceToImage(sum(RadIndividualBounceScalar(:,:,:,2:end) + RadIndividualBounceVector(:,:,:,2:end), 4), Scene, Cam);
ITotal      = ConvertRadianceToImage(sum(RadIndividualBounceScalar + RadIndividualBounceVector, 4), Scene, Cam);

% imwrite(im2uint16(imresize(IDirect,1)), [outdir, '/Image-Direct', '-Phase-', sprintf('%03d', CamPhase), '.tiff'])
% imwrite(im2uint16(imresize(IGlobal,1)), [outdir, '/Image-Global', '-Phase-', sprintf('%03d', CamPhase), '.tiff'])
imwrite(im2uint16(imresize(ITotal,1)), [outdir, '/Image-Total', '-Phase-', sprintf('%03d', CamPhase), '.tiff'])
