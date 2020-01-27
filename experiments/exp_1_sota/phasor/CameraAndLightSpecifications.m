
% Script specifying the camera and light parameters. Coordinate system is: 
% X-axis (left->right), Y-axis(bottom->top), Z-axis(in->out).
% Suppose the camera+light are on the Z-axis, looking towards the origin
% (inside the screen). Both the light source and the camera are assumed to
% be points (perspective projection). 
%
% Output are the structures Light and Camera. 
% 
% Fields of Light are Pos (position), I0 (strength), Freqs (frequencies), 
% Offsets (offsets), Amps (amplitudes). It is assumed that the phase of 
% each frequency is zero. 
% 
% Fields of Camera are Pos (position), LuxToPhoton (conversion factor
% between lux and photons), ExposureTime (seconds), ReadNoise (standard
% deviation in electrons), FullWellCap (electrons), CameraGain (number of
% photons per grey level), NumBits (bit-depth). Also, Freqs (frequencies), 
% Offsets (offsets), Amps (amplitudes) for the reference signal (using
% either electronics or external modulation). 



%%%%%%%%%%%%%%%%%%%%%%%%%% Light Specifications %%%%%%%%%%%%%%%%%%%%%%%%%%%
Light.Pos                   = [0, 0, 4.5];                                                                                                                          % meters
Light.I0                    = 20e3;                                                                                                                                 % lux / second
Light.Freqs                 = [1063.3, 1034.1] * 1.0e6;                                                                                                             % hertz
Light.Offsets               = 0.5 * ones(size(Light.Freqs));                                                                                                        % Assume light signal is normalized sinusoid (value between 0 and 1)
Light.Amps                  = 0.5 * ones(size(Light.Freqs));                                                                                                        % Assume light signal is normalized sinusoid (value between 0 and 1)


%%%%%%%%%%%%%%%%%%%%%%%%%% Camera Specifications %%%%%%%%%%%%%%%%%%%%%%%%%%
% Geometric (extrinsic)
Cam.Center                	= Light.Pos;                % Camera center (distance units are in meters)
Cam.DownVector           	= [0.0 -1.0 0.0];           % Down vector (not necessarily unit - will be unit-ified later)
Cam.LookAtVector         	= [0.0  0.0 -1.0];          % Look at vector (not necessarily unit - will be unit-ified later)
% Resolution 
Cam.NRow                 	= 2000;                   	% Image Resolution
Cam.NCol                  	= 2000;
% Intrinsic 
Cam.FocalLength           	= 900;                  	% In pixels
Cam.PrincipalPoint        	= [1000 1000];
% Pixel-size
Cam.PixSize             	= 2.5e-6;               	% in meters
% Noise, exposure, gain, LuxToPhoton, FullWellCap, etc.
Cam.Pos                     = Light.Pos;
Cam.LuXToPhoton          	= 10000;                                                % Convert lux value into number of photons (or electrons --- assume quantum efficiency = 1)
Cam.ExposureTime          	= 0.03;                                                 % Seconds
Cam.ReadNoise               = 10;                                                   % In electrons (standard deviation)
Cam.FullWellCap         	= 500000;
Cam.Gain                    = 10;                                                   % Number of photons per grey level
Cam.NumBits               	= 14;

Cam.Freqs                   = Light.Freqs;            
Cam.Offsets                 = 0.5 * ones(size(Cam.Freqs));                          % Assume reference signal is normalized sinusoid (value between 0 and 1)
Cam.Amps                    = 0.5 * ones(size(Cam.Freqs));                          % Assume reference signal is normalized sinusoid (value between 0 and 1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% Making other computed parameters %%%%%%%%%%%%%%%%%%%%
[KMatrix, RMatrix, TMatrix] = MakeCameraMatrix(Cam);
Cam.KMatrix                 = KMatrix;
Cam.RMatrix                 = RMatrix;
Cam.TMatrix                 = TMatrix;

clear KMatrix RMatrix TMatrix