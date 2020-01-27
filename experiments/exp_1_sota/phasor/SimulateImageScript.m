
clear; clc

%%%%%%%%%%%%%%%%%%%%%%% Declaring parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% output directory (where result images are stored)
baseDir         = './Images';

% shape ID (choose 1 for V-groove, 2 for Cornell box)
ShapeID         = 2;                    

% shape resolution (discretization for finite element scene description. smaller value -> higher resolution, more memory requirement). 
ShapeRes        = 0.011;

% Number of bounces simulated for the global component
NumGlobalBounces= 3;            


%%%%%%%%%%%%%%%%%%%%%%%%%%% Making images %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CameraAndLightSpecifications;                                       % script specifying imaging and illumination  (including modulation frequencies)
Scene           = SceneSpecifications(ShapeID, ShapeRes);           % function specifying scene. The second argument defines the spatial resolution of the scene. 

for freqNum=1:numel(Light.Freqs)                    % for every modulation frequency
    for Shift=[0:90:270]                            % phase-shifts
        
        ComputeIndividualBounceImagesVectorPieceWise(Scene, Light, Cam, NumGlobalBounces, freqNum, Shift, [baseDir, '/Set-0', sprintf('%02d', freqNum), '-Vector'], size(Scene.X, 2));
     
    end
end