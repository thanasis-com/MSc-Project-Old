totalfalnrate=0;
totalfalprate=0;
totalJacc=0;
for q=[1,2,3,7,10]
%% get parameters
all_parameters = setParameters();

filetoopen= ['images\October25_2013_Set_Labelled\',num2str(q),'.jpg'];
all_parameters.I = imread(filetoopen);

% image name
I = all_parameters.I;

% forbidden line
r = all_parameters.r; % ratio between the size of forbidden lines and 
                      % the size of the image 
                      
% radial symmetry response
maxRadius = all_parameters.maxRadius;  % maximum radius
minRadius = all_parameters.minRadius;  % minimum radius
radii = all_parameters.radii; % radii search range
alp = all_parameters.alp;  % radial strictness
gau = all_parameters.gau;  % size of Gaussian filter
p1th = all_parameters.p1th;  % threshold of symmetry responses

% non-maximum suppression
d = all_parameters.d;  % size of search region

% eliminate dark blobs
thI = all_parameters.thI;  % threshold of pixel values

% active contour segmentation
treCols = all_parameters.treCols;
ale = all_parameters.ale; % directions of the search space
lambda = all_parameters.lambda;  % weight of internal energy
Weight = all_parameters.Weight;  % weight of the directional gradient
contour_r = all_parameters.contour_r; % sample range along the directions of the search space

%% Nanodisc detection

rows=1024;
cols=1024;

% Region of forbidden lines
forbidWid = round(r*cols);
forbidHei = round(r*rows);
forbidWidS = floor((cols-forbidWid)/2)+1;
forbidWidE = forbidWidS+forbidWid-1;
forbidHeiS = floor((rows-forbidHei)/2)+1;
forbidHeiE = forbidHeiS+forbidHei-1;    
    
%% evaluation 

% specify the ground truth file
allimages = q;  % image number
ChrixmlFile = ['groundTruth\xml Labelled\Christian\', int2str(allimages), '.xml'];
JohnxmlFile = ['groundTruth\xml Labelled\John\', int2str(allimages), '.1.xml'];
xmlFile = JohnxmlFile;
pxmlFile = ChrixmlFile;

% forbidden line region
forbiddenLines.forbidWidS = forbidWidS;
forbiddenLines.forbidWidE = forbidWidE;
forbiddenLines.forbidHeiS = forbidHeiS;
forbiddenLines.forbidHeiE = forbidHeiE;

% read ground truth nanodiscs
[N, grTru, allPolygon, radii, polyCen, polyArea] = NewXMLReader( rows, cols, I, xmlFile, forbiddenLines );

% read test nanodiscs
[pN, pgrTru, pallPolygon, pradii, ppolyCen, ppolyArea] = NewXMLReader( rows, cols, I, pxmlFile, forbiddenLines );

% ground truth parameters
groundTruthPara.N = N;
groundTruthPara.grTru = grTru;
groundTruthPara.allPolygon = allPolygon;
groundTruthPara.polyCen = polyCen;

% calculate jaccard index
[jaccard_value, num_right] = jaccardEvaluationMutated( groundTruthPara, pgrTru, I, rows, cols, round(ppolyCen(:,1)), round(ppolyCen(:,2)), forbiddenLines );

% calculate false positives and false negatives per 100 nanodiscs
numCir = length(polyCen(:,1));  % number of all the segmentations
[falnrate, falprate] = falseRates( N, num_right, numCir );

totalJacc=totalJacc+mean(jaccard_value);
totalfalnrate=totalfalnrate+falnrate;
totalfalprate=totalfalprate+falprate;
end

meanTotalJacc=totalJacc/5;
meanfalprate=totalfalprate/5;
meanfalnrate=totalfalnrate/5;