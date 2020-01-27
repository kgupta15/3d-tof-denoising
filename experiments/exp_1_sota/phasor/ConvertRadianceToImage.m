
% This function converts scene radiance into image intensities by applying
% (1) perspective projection, and (2) proper image noise. 
% 
% Input: Matrix of radiance values RadMat, structure Cam, structure Scene
%
% Output: Matrix of image values IMat, same size as RadMat

function IMat   = ConvertRadianceToImage(RadMat, Scene, Cam)

%%%%%%%%%%%%%%%%%%% Applying perspective projection %%%%%%%%%%%%%%%%%%%%%%%

% Taking care of self shadowing
SceneCamDists                   = sqrt((Scene.X-Cam.Center(1)).^2 + (Scene.Y-Cam.Center(2)).^2 + (Scene.Z-Cam.Center(3)).^2);                                                       % Distance between scene elements and camera center
cosCamAngles                    = ((Cam.Center(1) - Scene.X) .* Scene.NX + (Cam.Center(2) - Scene.Y) .* Scene.NY + (Cam.Center(3) - Scene.Z) .* Scene.NZ) ./ SceneCamDists;         % cos of angles between scene normal and line joining scene point and camera
for i=1:size(RadMat, 3)
    RadMatTmp                   = RadMat(:,:,i);
    RadMatTmp(cosCamAngles<0)   = 0;                                                                                                                                                % Scene points that are self shadowed should be imaged as zero.
    RadMat(:,:,i)               = RadMatTmp;
end

% Projecting the scene onto the camera image
[CamImCols, CamImRows]    	= meshgrid([1:Cam.NCol],[1:Cam.NRow]);                                                                  % Grid of camera image pixel co-ordinates
CamMatrix                   = Cam.KMatrix * [Cam.RMatrix Cam.TMatrix];                                                              % 3X4 camera matrix

CamImCoordinates            = CamMatrix * [Scene.X(:)' ; Scene.Y(:)' ; Scene.Z(:)' ; ones(1,numel(Scene.X))];                       % Camera image co-ordinates corresponding to the scene points

CamImCoordinates(1,:)       = CamImCoordinates(1,:) ./ CamImCoordinates(3,:);                                                       % De-homogenizing the coordinates
CamImCoordinates(2,:)       = CamImCoordinates(2,:) ./ CamImCoordinates(3,:);        

CamImCoordinatesCol         = reshape(CamImCoordinates(1,:), size(Scene.X));                                                        % Reshaping the projected coordinates to be the same size as the scene matrix
CamImCoordinatesRow         = reshape(CamImCoordinates(2,:), size(Scene.X));

% Interpolating to compute the irradiance
IrrMat                    	= zeros(Cam.NRow, Cam.NCol, 3);

for i=1:size(RadMat, 3)               % for each color channel
    if(Scene.ShapeNum==1)               % For v-groove, split into two because of scattered interpolation issues
        NumPieces               = 2;
        BreakIndices            = [0, floor(size(RadMat,2)/2), size(RadMat,2)];
    elseif(Scene.ShapeNum==2)           % For cornell-box, split into five because of scattered interpolation issues
        NumPieces               = 5; 
        BreakIndices            = [0, floor(size(RadMat,2)/5), floor(size(RadMat,2)*2/5), floor(size(RadMat,2)*3/5), floor(size(RadMat,2)*4/5), size(RadMat,2)];
    end
    
    for sh=1:NumPieces
        SceneRadiances          = RadMat(:,BreakIndices(sh)+1:BreakIndices(sh+1),i);
        CamImCoordinatesColTmp 	= CamImCoordinatesCol(:, BreakIndices(sh)+1:BreakIndices(sh+1));
        CamImCoordinatesRowTmp	= CamImCoordinatesRow(:, BreakIndices(sh)+1:BreakIndices(sh+1));
        FInterpolant         	= TriScatteredInterp([CamImCoordinatesColTmp(:), CamImCoordinatesRowTmp(:)], SceneRadiances(:), 'natural');            % Creating the interpolant function
        ITmp                    = FInterpolant(CamImCols, CamImRows);
        ITmp((isnan(ITmp)))     = 0;
        IrrMat(:,:,i)         	= IrrMat(:,:,i) + ITmp;
    end
end


%%%%%%%% So far, IMat contains lux values in image coordinates %%%%%%%%%%%%
%%%%%%%%%%% Now, converting them to grey levels, after adding  %%%%%%%%%%%%
%%%%%%%%%%% noise, applying gain, saturation, full-well-capacity, etc. %%%%

PhotonTotal                 = IrrMat * Cam.LuXToPhoton;                                                               	% Convert lux value into number of photons
NumPhotons                  = PhotonTotal * Cam.ExposureTime;                                                           % Number of photons captured per exposure
Noise                       = sqrt(NumPhotons) .* randn(size(NumPhotons)) + Cam.ReadNoise * randn(size(NumPhotons));   	% Noise value for each frame (two different randn because two different random processes)

IMat                        = min(NumPhotons + Noise, Cam.FullWellCap);                                                 % Apply full well capacity
IMat(IMat<0)                = 0;

% Applying gain, and quantization
IMat                        = IMat / Cam.Gain;                                                                          % Applying camera gain
IMat                        = round(IMat) / 2^Cam.NumBits;                                                              % Applying quantization. Applying scaling to account for the bit-depth