disp('Setting up the tools for todays exercise in 02931...');

% Setup path
path=pwd;
if strcmp(path(1),'/') % linux system
    addpath([path '/FastICA_25'])
    addpath([path '/Data'])
    addpath([path '/Tools'])   
    addpath([path '/Solutions'])   
else % Windows system       
    addpath([path '\FastICA_25'])
    addpath([path '\Data'])
    addpath([path '\Tools'])       
    addpath([path '\Solutions'])   
end
clear path;

% Chech version number
v = ver('MATLAB');
verVec = sscanf(v.Version, '%d.%d.%d');
verNum = sum(verVec.*logspace(0, 1-length(verVec), length(verVec))');
fprintf('Running Matlab version %s\n',v.Version);

% Done
disp('Setup of the 02582 tools for this exercise completed.');

