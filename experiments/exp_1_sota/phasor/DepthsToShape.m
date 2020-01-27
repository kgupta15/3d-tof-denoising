% This function takes the camera matrix and the depths computed at each
% pixel and returns the Shape structure (with X, Y, Z sub-structures). 
% The shape is computed in the camera coordinate system, not the world
% coordinate system. World coordinates can be computed by applying camera's
% rotation and translation matrices. 


function Shape = DepthsToShape(Cam, Depths)

[PX, PY]    = meshgrid([1:Cam.NCol], [1:Cam.NRow]);                         % Pixel coordinates (X and Y)
PX          = PX - Cam.PrincipalPoint(2);                                   % Subtracting the principal point coordinates 
PY          = PY - Cam.PrincipalPoint(1); 
PZ          = Cam.FocalLength * ones(size(PX)); 
PixDists    = sqrt(PX.^2 + PY.^2 + PZ.^2);

% Normalizing the coordinates to make a unit vector
PX          = PX ./ PixDists;
PY          = PY ./ PixDists;
PZ          = PZ ./ PixDists;

Shape.X     = PX .* Depths;
Shape.Y     = PY .* Depths;
Shape.Z     = PZ .* Depths;