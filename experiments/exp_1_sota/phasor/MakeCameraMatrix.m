
% This function computes the 3X4 camera matrix, given the camera
% parameters. 
% 
% Input: Cam structure
%
% Outputs:
% 1) KMatrix : Internal parameters matrix (3X3)
% 2) RMatrix : Rotation matrix (3X3)
% 3) TMatrix : Translation matrix (3X1)

function [KMatrix, RMatrix, TMatrix] = MakeCameraMatrix(Cam)

% Making KMatrix

KMatrix     = [ Cam.FocalLength      0        Cam.PrincipalPoint(1);
                 0          Cam.FocalLength   Cam.PrincipalPoint(2);
                 0               0                 1];             %% Assuming no pixel skew and assuming square pixels
             

%%% Making RMatrix and TMatrix

ZVec        = [Cam.LookAtVector(1) Cam.LookAtVector(2) Cam.LookAtVector(3)]';
YVec        = [Cam.DownVector(1)   Cam.DownVector(2)   Cam.DownVector(3)]';
XVec        = cross(YVec, ZVec);

RCamToWorld = [XVec YVec ZVec];     %% Rotation matrix from the camera co-ordinate system to the world co-ordinate system
RWorldToCam = RCamToWorld';         %% Rotation matrix from the camera co-ordinate system to the world co-ordinate system {P_W = R*P_C + T --> P_C = R'*P_W - R'*T  (Camera center = -R'' * (-R'*T) = T)}
TWorldToCam = -RCamToWorld' * [Cam.Center(1) Cam.Center(2) Cam.Center(3)]';

RMatrix     = RWorldToCam;
TMatrix     = TWorldToCam;