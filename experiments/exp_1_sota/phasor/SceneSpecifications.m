
% Function specifying the scene parameters. Coordinate system is: X-axis (left->right), Y-axis(bottom->top), Z-axis(in->out)
% Suppose the camera is on the Z-axis, looking towards the origin (inside
% the screen)
%
% Inputs:
% 1) ShapeID: 1 for V-groove, 2 for Cornell box
% 2) ShapeRes: StepSize for finite element discretization. Smaller the
% value, higher the resolution. Typical values: [0.005:0.05].
%
% Output: Structure Scene with fields X, Y, Z, NX, NY, NZ, Areas,
% Albedos, Roughness, FresnelTerm


function Scene     = SceneSpecifications(ShapeID, ShapeRes)

if(ShapeID==1)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Define V-groove %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % The fold of V-groove is along Y axis. The V-groove is symmetric about the X-Z plane.
    
    ApexAngleLeft   = 20;               % degrees (angle between the left plane and the y-z plane)
    ApexAngleRight  = 50;               % degrees (angle between the right plane and the y-z plane)
    LengthAxis      = 4.0;              % Length along the axis (meters)
    LengthLeft      = 3.0;              % Length of the left face (meters)
    LengthRight     = 3.0;           	% Length of the right face (meters)
    dFinite         = ShapeRes;      	% discrete element size along all three dimensions
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % LeftFace
    XLeftTmp        = -[LengthLeft:-dFinite:0]*sin(ApexAngleLeft * pi/180);
    ZLeftTmp        = [LengthLeft:-dFinite:0]*cos(ApexAngleLeft * pi/180);
    YLeftTmp        = [-LengthAxis/2:dFinite:LengthAxis/2];
    
    XLeft           = repmat(XLeftTmp, [numel(YLeftTmp) 1]);
    ZLeft           = repmat(ZLeftTmp, [numel(YLeftTmp) 1]);
    YLeft           = repmat(YLeftTmp', [1 numel(ZLeftTmp)]);
    
    
    % RightFace
    XRightTmp     	= [dFinite:dFinite:LengthRight]*sin(ApexAngleRight * pi/180);
    ZRightTmp     	= [dFinite:dFinite:LengthRight]*cos(ApexAngleRight * pi/180);
    YRightTmp      	= [-LengthAxis/2:dFinite:LengthAxis/2];
    
    XRight        	= repmat(XRightTmp, [numel(YRightTmp) 1]);
    ZRight        	= repmat(ZRightTmp, [numel(YRightTmp) 1]);
    YRight        	= repmat(YRightTmp', [1 numel(ZRightTmp)]);
    
    
    % Merging the two faces
    X               = [XLeft, XRight];
    Y               = [YLeft, YRight];
    Z               = [ZLeft, ZRight];
    
    clear XLeft XRight YLeft YRight ZLeft ZRight
    
    
    % Computing normals
    
    NormalLeft      = [sin(pi/2 - ApexAngleLeft * pi/180), 0, cos(pi/2 - ApexAngleLeft * pi/180)];
    NormalRight   	= [-sin(pi/2 - ApexAngleRight * pi/180), 0, cos(pi/2 - ApexAngleRight * pi/180)];
    
    NXLeft          = NormalLeft(1) * ones(numel(YLeftTmp), numel(XLeftTmp));
    NYLeft          = NormalLeft(2) * ones(numel(YLeftTmp), numel(XLeftTmp));
    NZLeft          = NormalLeft(3) * ones(numel(YLeftTmp), numel(XLeftTmp));
    
    NXRight        	= NormalRight(1) * ones(numel(YRightTmp), numel(XRightTmp));
    NYRight       	= NormalRight(2) * ones(numel(YRightTmp), numel(XRightTmp));
    NZRight       	= NormalRight(3) * ones(numel(YRightTmp), numel(XRightTmp));
    
    NX              = [NXLeft, NXRight];
    NY              = [NYLeft, NYRight];
    NZ              = [NZLeft, NZRight];
    
    clear NormalLeft NormalRight
    clear XLeftTmp ZLeftTmp YLeftTmp
    clear XRightTmp ZRightTmp YRightTmp
    clear NXLeft NXRight NYLeft NYRight NZLeft NZRight
    
    
    % Computing areas
    LengthsHorizontal   = sqrt((circshift(X, [0 -1 0]) - X).^2 + (circshift(Y, [0 -1 0]) - Y).^2 + (circshift(Z, [0 -1 0]) - Z).^2);
    LengthsVertical     = sqrt((circshift(X, [-1 0 0]) - X).^2 + (circshift(Y, [-1 0 0]) - Y).^2 + (circshift(Z, [-1 0 0]) - Z).^2);
    Areas               = LengthsHorizontal .* LengthsVertical;
    Areas(end,:)        = Areas(end-1,:);                           % correcting the boundaries
    Areas(:,end)        = Areas(:, end-1);
    
    % Computing albedos
    Albedos                                             = zeros(size(X,1), size(X,2), 3);
    Albedos(:,1:floor(size(X,2)/2),1)                   = 1;
    Albedos(:,1:floor(size(X,2)/2),2)                   = 0;
    Albedos(:,1:floor(size(X,2)/2),3)                   = 0;
    Albedos(:,floor(size(X,2)/2)+1:size(X,2),1)         = 1;
    Albedos(:,floor(size(X,2)/2)+1:size(X,2),2)         = 1;
    Albedos(:,floor(size(X,2)/2)+1:size(X,2),3)         = 0;
    
    clear ApexAngleLeft ApexAngleRight LengthAxis LengthLeft LengthRight dFinite Albedo LengthsHorizontal LengthsVertical
    
    
    % Making the structure
    Scene.X             = X;
    Scene.Y             = Y;
    Scene.Z             = Z;
    Scene.NX            = NX;
    Scene.NY            = NY;
    Scene.NZ            = NZ;
    Scene.Areas         = Areas;
    Scene.Albedos    	= Albedos;
    Scene.ShapeNum      = 1;                        % Shape identifier. If 1, then VGroove. If 2, then CornellBox.
    
    clear X Y Z NX NY NZ Areas AlbedosR AlbedosG AlbedosB Albedos
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    
elseif(ShapeID==2)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Define Cornell Box %%%%%%%%%%%%%%%%%%%%%%%%%%
    % The back face of the box is on the X-Z plane. Assuming that each face is
    % a square.
    
    LengthSide          = 3.0;              % Length of each side (assuming square faces)
    dFinite             = ShapeRes;         % discrete element size along all three dimensions
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % LeftFace
    [ZLeft, YLeft]      = meshgrid([LengthSide:-dFinite:0], [LengthSide/2:-dFinite:-LengthSide/2]);
    XLeft               = -LengthSide/2 * ones(size(ZLeft));
    
    % BackFace
    [XBack, YBack]   	= meshgrid([-LengthSide/2:dFinite:LengthSide/2], [LengthSide/2:-dFinite:-LengthSide/2]);
    ZBack               = zeros(size(XBack));
    
    % RightFace
    [ZRight, YRight] 	= meshgrid([0:dFinite:LengthSide], [LengthSide/2:-dFinite:-LengthSide/2]);
    XRight           	= LengthSide/2 * ones(size(ZRight));
    
    % TopFace
    [XTop, ZTop]        = meshgrid([-LengthSide/2:dFinite:LengthSide/2], [LengthSide:-dFinite:0]);
    YTop                = LengthSide/2 * ones(size(XTop));
    
    % BottomFace
    [XBottom, ZBottom] 	= meshgrid([-LengthSide/2:dFinite:LengthSide/2], [LengthSide:-dFinite:0]);
    YBottom             = -LengthSide/2 * ones(size(XBottom));
    
    % Merging the faces
    X               = [XLeft, XBack, XRight, XTop, XBottom];
    Y               = [YLeft, YBack, YRight, YTop, YBottom];
    Z               = [ZLeft, ZBack, ZRight, ZTop, ZBottom];
    
    % Computing normals
    NX              = [ones(size(XLeft)), zeros(size(XBack)), -ones(size(XRight)), zeros(size(XTop)), zeros(size(XBottom))];
    NY              = [zeros(size(XLeft)), zeros(size(XBack)), zeros(size(XRight)), -ones(size(XTop)), ones(size(XBottom))];
    NZ              = [zeros(size(XLeft)), ones(size(XBack)), zeros(size(XRight)), zeros(size(XTop)), zeros(size(XBottom))];
    
    % Computing albedos
    Albedos(:,:,1)  = [ones(size(XLeft)), ones(size(XBack)), zeros(size(XRight)), ones(size(XTop)), ones(size(XBottom))];
    Albedos(:,:,2)  = [zeros(size(XLeft)), ones(size(XBack)), ones(size(XRight)), ones(size(XTop)), ones(size(XBottom))];
    Albedos(:,:,3)  = [zeros(size(XLeft)), ones(size(XBack)), zeros(size(XRight)), ones(size(XTop)), ones(size(XBottom))];
    
    % Computing areas
    Areas           = dFinite^2 * ones(size(X));
    
    clear XLeft XBack XRight XTop XBottom YLeft YBack YRight YTop YBottom ZLeft ZBack ZRight ZTop ZBottom
    
    % Making the structure
    Scene.X             = X;
    Scene.Y             = Y;
    Scene.Z             = Z;
    Scene.NX            = NX;
    Scene.NY            = NY;
    Scene.NZ            = NZ;
    Scene.Areas         = Areas;
    Scene.Albedos    	= Albedos;
    Scene.ShapeNum      = 2;                        % Shape identifier. If 1, then VGroove. If 2, then CornellBox.
    
    clear X Y Z NX NY NZ Areas Albedos
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end