function all_parameters = setParameters()
%Set paramters for the nanodisc detection
% Returns:
%    all_parameters: most important parameters in the algorithm


% image name
%all_parameters.I = imread('imgBig46.png');

% forbidden line
all_parameters.r = 0.8;  % ratio between the size of forbidden lines and 
                         % the size of the image 

% radial symmetry response
all_parameters.maxRadius = 100;  % maximum radius50 64without
all_parameters.minRadius = 25;  % minimum radius24 20without
all_parameters.radii = all_parameters.minRadius:2:all_parameters.maxRadius; % radii search range
all_parameters.alp = 1;  % radial strictness
all_parameters.gau = 6;  % size of Gaussian filter
all_parameters.p1th = 0.09;  % threshold of symmetry responses

% non-maximum suppression
all_parameters.d = all_parameters.minRadius;  % size of search region

% eliminate dark blobs
all_parameters.thI = 20;  % threshold of pixel values


% active contour segmentation
all_parameters.treCols = 50;
all_parameters.ale = 0:2*pi/all_parameters.treCols:(2*pi-2*pi/all_parameters.treCols); % directions of the search space
all_parameters.lambda = 0.95;  % weight of internal energy 0.95
all_parameters.Weight = 1;  % weight of the directional gradient 1

all_parameters.contour_r = 1; % sample range along the directions of the search space


end

