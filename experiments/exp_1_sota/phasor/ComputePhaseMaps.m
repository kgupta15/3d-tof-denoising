
% This function computes the phase-maps for a given frequency, and a set of
% phase-shifted images, and the phase shift vector.

function [PhaseMap]  =  ComputePhaseMaps(IMat, shiftVec)

nr          = size(IMat, 1);
nc          = size(IMat, 2);
NumShifts   = numel(shiftVec);

%%%% Now solving the frequency equations
A   = [ones(NumShifts,1)    cos(shiftVec)'   -sin(shiftVec)'];          %%% Assumes that the images are formed using cos(phi - delta)
B   = reshape(IMat, [nr*nc NumShifts])';
C   = A\B;
clear A B

Amp                     = sqrt(C(2,:).^2 + C(3,:).^2);
Phase                   = acos(C(2,:)./Amp);
Phase((C(3,:)<0))       = 2*pi - Phase((C(3,:)<0));
PhaseMap                = reshape(Phase, [nr nc]);

clear C Phase Amp