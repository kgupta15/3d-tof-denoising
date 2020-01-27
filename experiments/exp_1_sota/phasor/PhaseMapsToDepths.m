
% This function computes the final depth, given individual phases and the
% vector of frequencies. Also, given as input is the depth range - which is
% the set of possible depth values. 

function [Depths]   = PhaseMapsToDepths(freqVec, PhaseMaps, DepthRange)

nr      = size(PhaseMaps, 1);
nc      = size(PhaseMaps, 2);
nFreq   = size(PhaseMaps, 3);

% Computing the phases for every candidate depth
CandidatePhases     = zeros(1, numel(DepthRange), nFreq);
for i=1:nFreq
    CandidatePhases(1,:,i)      = mod(2* pi * 2 * DepthRange / (3e11/freqVec(i)), 2*pi);       % Multiply depth range by 2 because light traverses the distance twice
end
CandidatePhases     = repmat(CandidatePhases, [nr 1 1]);

% Computing the depths
Depths  = zeros(nr, nc);

for i=1:nc                  % Consider one column at a time
    i
    
    PhaseMapsTmp    = PhaseMaps(:,i,:);
    PhaseMapsTmp    = repmat(PhaseMapsTmp, [1 numel(DepthRange) 1]);
    
    ErrMat          = sum((PhaseMapsTmp - CandidatePhases).^2, 3);
%     keyboard
    [~, indices]   	= min(ErrMat, [], 2);
    
    Depths(:,i)     = DepthRange(indices);
end