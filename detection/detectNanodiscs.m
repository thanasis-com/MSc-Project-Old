function [] = detectNanodiscs(originalImg, filename)
%% get parameters
all_parameters = setParameters();

originalImage=rgb2gray(imread(['..\..\images\',originalImg]));

filetoopen= filename;
all_parameters.I =padarray(imcomplement(imread(filetoopen)),[103 103],'both');

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

%I = rgb2gray(I);
[rows,cols]=size(I);

% Region of forbidden lines
forbidWid = round(r*cols);
forbidHei = round(r*rows);
forbidWidS = floor((cols-forbidWid)/2)+1;
forbidWidE = forbidWidS+forbidWid-1;
forbidHeiS = floor((rows-forbidHei)/2)+1;
forbidHeiE = forbidHeiS+forbidHei-1;    

% Gaussian smoothing
im = gaussfilt(gau, gau*0.2, I);

% Compute derivatives in x and y via Farid and Simoncelli's 5 tap
% derivative filters
[imgx, imgy] = derivative5(im, 'x', 'y');   
mag = sqrt(imgx.^2 + imgy.^2)+eps; % (+eps to avoid division by 0)    
beta2 = max(mag(:));

copyimgx = imgx;
copyimgy = imgy;

% Normalise gradient values so that [imgx imgy] form unit 
% direction vectors.
imgx = imgx./mag;   
imgy = imgy./mag;
   
% Threshold to ignore small gradients
magth = max(mag(:))*0.02;

p1 = zeros(rows,cols);  % symmetry response map
p2 = zeros(rows,cols);  % corresponding radius map

[x,y] = meshgrid(1:cols, 1:rows);


%% An implementation of Loy and Zelinski's fast radial feature detector
% This code is adapted from Peter Kovesi's code
% Copyright (c) 2004-2010 Peter Kovesi
% Centre for Exploration Targeting
% The University of Western Australia
% http://www.csse.uwa.edu.au/~pk/research/matlabfns/

for n = radii

    % Coordinates of 'positively' and 'negatively' affected pixels
    negx = x - round(n*imgx);
    negy = y - round(n*imgy);

    % M: Magnitude projection image
    % O: Orientation projection image    
    [O, M] = projectionMap(rows, cols, mag, magth, negx, negy);


    % Unsmoothed symmetry measure at this radius value
    % We use the radius 'n' as the normalization parameter instead of a
    % fixed normalization parameter used in Loy and Zelinski's paper.
    F = M./n .* (abs(O)./n).^alp; 

    ker = ones(1, n);  % size of the kernel used to gather symmetry responses    
    F = conv2(F, ker, 'same');
    F = conv2(F, ker', 'same');

    p2(F<p1) = n;
    p1(F<p1) = F(F<p1);

end        

p1 = abs(p1);

% normalize p1 to range [0 1]
maxp1 = max(p1(:));
minp1 = min(p1(:));
p1 = (p1-minp1)./(maxp1-minp1);

fprintf('Radial symmetry detection is done.\n\n');
    
%% ignore the points which have small symmetry response
nonMax = zeros(rows,cols);
nonMax(p1>p1th) = 1;   
index = find(nonMax == 1);    


%% Eliminate black blobs
lnonMax1 = length(index);
for i = 1:lnonMax1
    if I(index(i)) < thI 
        nonMax(index(i)) = 0;
    end
end    
[y1,x1] = find(nonMax == 1);



%% Only keep the nanodiscs in the forbidden line
alpha=0:pi/20:2*pi; % [0,2*pi]
index = (x1-1).*rows+y1;
numCir = length(index);
for i = 1:numCir
    R = p2(index(i));
    x=R*cos(alpha);
    y=R*sin(alpha);
    x = x+x1(i);
    y = y+y1(i);
    if ~isempty(find(x>forbidWidE,1)) || ~isempty(find(y<forbidHeiS,1)) ...
            || ~isempty(find(x<forbidWidS,1)) || ~isempty(find(y>forbidHeiE,1))
        
        nonMax(index(i)) = 0;        
    
    end
   
end
[y1,x1] = find(nonMax == 1);



%% non-maximum suppression
x1L = length(x1);
for i = 1:x1L
    if y1(i)-d<1
        whStart = 1;
    else
        whStart = y1(i)-d;
    end
    if y1(i)+d>rows
        whEnd = rows;
    else
        whEnd = y1(i)+d;
    end
    if x1(i)-d<1
        wwStart = 1;
    else
        wwStart = x1(i)-d;
    end
    if x1(i)+d>cols
        wwEnd = cols;
    else
        wwEnd = x1(i)+d;
    end
    A = p1(whStart:whEnd, wwStart:wwEnd);  % the region for non-maximum suppression
    if ~isempty(find(A>p1(y1(i),x1(i)), 1))
        nonMax(y1(i),x1(i)) = 0;
    end
end

[y1,x1] = find(nonMax == 1);

  

%% Elimilate detections based on the distance between centers
index = find(nonMax == 1);
allRadii = p2(index);
allRes = p1(index);
res = [x1, y1, allRadii, allRes];
res = sortrows(res,-4);  % sort the detections by their symmetry responses

sign = 0;
i = 2;
while(i <= size(res,1))
    for j = 1:i-1
        d = sqrt((res(i,1)-res(j,1))^2+(res(i,2)-res(j,2))^2);
        % if the distance between two detection centres is smaller than
        % their radii, ignore the detection which has smaller symmetry response
        if (d<res(j,3)) || (d<res(i,3))
            nonMax(res(i,2), res(i,1)) = 0;
            res(i,:) = [];
            sign = 1;
            break;
        end
    end    
    if sign == 0
        i = i+1;
    else
        sign = 0;
    end
end

x1 = res(:,1);
y1 = res(:,2);



%% Active contour segmentation 
imgx = copyimgx;
imgy = copyimgy;

aL = length(ale);
fL = length(y1);

finalPath = zeros(fL, treCols);  % store the final contours of all the nanodisc detections

for i=1:fL  % for each nanodisc detection

    R = p2(y1(i),x1(i));
    len = (R+contour_r*R):-1:(R-contour_r*R); % search region between inner
                                              % circle and outer circle
    treRows = length(len);
 
    for rep = 1:2  % repeat the active contour segmentation two times
        
        magTre = zeros(treRows, treCols);  % store the gradient magnitudes of sample points
        magXTre = zeros(treRows, treCols); % store the directional gradient magnitudes of sample points related to x coordinate
        magYTre = zeros(treRows, treCols); % store the directional gradient magnitudes of sample points related to y coordinate
        wholeIndex = zeros(treRows, treCols);  % store all the indices of sample points

        pathValues = zeros(treRows*treRows, treCols);  % store the minimum energies in Viterbi search 
        pathEdges = zeros(treRows*treRows, treCols);  % store the points which generate minimum energies in Viterbi search
        
        minPath = zeros(1, treCols);

        % find the sample points in each direction of the search space
        for j = 1:aL  % for each direction
            colIndex = round(len.*cos(ale(j)));
            rowIndex = round(len.*sin(ale(j)));
            colIndex = colIndex+x1(i);
            colIndex(colIndex<1)=1;
            colIndex(colIndex>cols)=cols;
            rowIndex = rowIndex+y1(i);
            rowIndex(rowIndex<1)=1;
            rowIndex(rowIndex>rows)=rows;
            totalIndex = (colIndex-1).*rows + rowIndex;
            
            % store the gradient magnitudes, directional gradient magnitudes
            % and indices for the sample points
            wholeIndex(:, j) = totalIndex;
            magTre(:, j) = mag(totalIndex);
            magXTre(:, j) = imgx(totalIndex).*cos(ale(j));
            magYTre(:, j) = imgy(totalIndex).*sin(ale(j));

        end
        
        % Use Viterbi algorithm to find the best path (segmentation) in the search space
        for stage = 2:(treCols-1)
            vi_1 = [ceil(wholeIndex(:, stage-1)./rows), rem(wholeIndex(:, stage-1), rows)];
            for three = 1:treRows
                vi1= [ceil(wholeIndex(three, stage+1)./rows), rem(wholeIndex(three, stage+1), rows)];
                for two = 1:treRows
                    vi = [ceil(wholeIndex(two, stage)./rows), rem(wholeIndex(two, stage), rows)];
             
                    if magXTre(two, stage)+magYTre(two, stage) <= 0
                        Eext1 = 0;
                    else
                        Eext1 =  magTre(two, stage) ;
                    end
                    % external energy
                    Eext = -(( magXTre(two, stage)+magYTre(two, stage) )*Weight + Eext1*(1-Weight));
                    
                    % internal energy
                    Eint = ((vi_1(:,1)-2*vi(1)+vi1(1)).^2+(vi_1(:,2)-2*vi(2)+vi1(2)).^2)...
                        ./ ((vi_1(:,1)-vi1(1)).^2+(vi_1(:,2)-vi1(2)).^2);

                    [Si, indexSi] = min( pathValues((two*treRows-treRows+1):two*treRows, stage-1)...
                        + lambda.*Eint + (1-lambda).*Eext );
                    
                    pathEdges((three-1)*treRows+two, stage) = (two-1)*treRows+indexSi;
                    pathValues((three-1)*treRows+two, stage) = Si;
                end
            end
            
        end
        [~, endS] = min(pathValues(:, treCols-1));
        
        % backward method to find the path which has the minimum energy
        minPath(treCols-1) = endS;
        for pathLen = treCols-2:-1:2
            minPath(pathLen) = pathEdges(minPath(pathLen+1), pathLen+1);
        end
        
        cenMinPath = mod(minPath(2:end-1), treRows);
        cenMinPath(cenMinPath==0) = cenMinPath(cenMinPath==0)+treRows;
        minPath(2:end-1) = cenMinPath;
        
        
        % second search, choosing two middle points in the first search
        % as the start and end points for a second search
        newStart = floor(treCols/2)+1; 
        newEnd = floor(treCols/2);
        minPath(1) = minPath(newStart);
        minPath(end) = minPath(newEnd);
        
        % rearrange the trellis
        wholeIndex = [wholeIndex(:, newStart:end), wholeIndex(:, 1:newEnd)];
        wholeIndex(:, 1) = wholeIndex(minPath(1), 1);
        wholeIndex(:, end) = wholeIndex(minPath(end), end);       
        
        magTre = [magTre(:, newStart:end), magTre(:, 1:newEnd)];
        magTre(:, 1) = magTre(minPath(1), 1);
        magTre(:, end) = magTre(minPath(end), end);  
        
        magXTre = [magXTre(:, newStart:end), magXTre(:, 1:newEnd)];
        magXTre(:, 1) = magXTre(minPath(1), 1);
        magXTre(:, end) = magXTre(minPath(end), end);  
        
        magYTre = [magYTre(:, newStart:end), magYTre(:, 1:newEnd)];
        magYTre(:, 1) = magYTre(minPath(1), 1);
        magYTre(:, end) = magYTre(minPath(end), end);  
        
        pathEdges = zeros(treRows*treRows, treCols);
        pathValues = zeros(treRows*treRows, treCols);
        
        % same with the first search
        for stage = 2:treCols
            vi_1 = [ceil(wholeIndex(:, stage-1)./rows), rem(wholeIndex(:, stage-1), rows)];
            for three = 1:treRows
                
                if stage == treCols
                    vi1= [ceil(wholeIndex(three, 1)./rows), rem(wholeIndex(three, 1), rows)];
                else
                    vi1= [ceil(wholeIndex(three, stage+1)./rows), rem(wholeIndex(three, stage+1), rows)];
                end
                
                for two = 1:treRows
                    vi = [ceil(wholeIndex(two, stage)./rows), rem(wholeIndex(two, stage), rows)];
             
                    if magXTre(two, stage)+magYTre(two, stage) <= 0
                        Eext1 = 0;
                    else
                        Eext1 =  magTre(two, stage) ;
                    end
                    Eext = -(( magXTre(two, stage)+magYTre(two, stage) )*Weight + Eext1*(1-Weight));
                    
                    Eint = ((vi_1(:,1)-2*vi(1)+vi1(1)).^2+(vi_1(:,2)-2*vi(2)+vi1(2)).^2)...
                        ./ ((vi_1(:,1)-vi1(1)).^2+(vi_1(:,2)-vi1(2)).^2);

                    [Si, indexSi] = min( pathValues((two*treRows-treRows+1):two*treRows, stage-1)...
                        + lambda.*Eint + (1-lambda).*Eext );
                    
                    pathEdges((three-1)*treRows+two, stage) = (two-1)*treRows+indexSi;
                    pathValues((three-1)*treRows+two, stage) = Si;
                end
            end
            
        end
        [~, endS] = min(pathValues(:, treCols));
        
        minPath(treCols-1) = pathEdges(endS, end);
        for pathLen = treCols-2:-1:2
            minPath(pathLen) = pathEdges(minPath(pathLen+1), pathLen+1);
        end
        
        cenMinPath = mod(minPath(2:end-1), treRows);
        cenMinPath(cenMinPath==0) = cenMinPath(cenMinPath==0)+treRows;
        minPath(2:end-1) = cenMinPath;              

        % get the final contours for each nanodisc
        for m = 1:treCols
            finalPath(i, m) = wholeIndex(minPath(m), m);
        end        
        
        % new centres
        indexx = ceil(finalPath(i, :)./rows);
        indexy = rem(finalPath(i, :), rows);  
        xi1 = [indexx(2:end), indexx(1)];
        yi1 = [indexy(2:end), indexy(1)];
        A = 0.5 * sum( indexx.*yi1 - xi1.*indexy );
        % calculate the centroid of the segmentation
        Cx = abs( sum( (indexx + xi1).*(indexx.*yi1 - xi1.*indexy) )./(6*A) );    % x-coordinate of the centroid
        Cy = abs( sum( (indexy + yi1).*(indexx.*yi1 - xi1.*indexy) )./(6*A) );   % y-coordinate of the centroid  
        x1(i) = round(Cx);
        y1(i) = round(Cy);
        res(i,1:2) = [x1(i), y1(i)];
        
    end
    
    fprintf('nanodisc %d is completed, %d left.\n', i, fL-i);

end

x1 = res(:,1);
y1 = res(:,2);

% clear negx negy F cenSum


%% Only keep the nanodiscs in the forbidden line
index = (x1-1).*rows+y1;
i = 1;
while(i <= length(index))
    x = ceil(finalPath(i, :)./rows);
    y = rem(finalPath(i, :), rows);
    if ~isempty(find(x>forbidWidE,1)) || ~isempty(find(y<forbidHeiS,1)) ...
            || ~isempty(find(x<forbidWidS,1)) || ~isempty(find(y>forbidHeiE,1))
       
        finalPath(i, :) = [];
        index(i) = [];
        res(i, :) = [];         
    else
        i = i+1;
    end        
end

x1 = res(:,1);
y1 = res(:,2);


%% Elimilate overlapping segmentations
regionX = repmat(1:cols,rows,1);
regionY = repmat((1:rows)', 1, cols);

index = (x1-1).*rows+y1;
viterbiRegion = zeros(rows, cols);
sign = 0;
i = 2;
while(i <= length(index))  
    x = ceil(finalPath(i-1, :)./rows);
    y = rem(finalPath(i-1, :), rows);
    IN = inpolygon(regionX,regionY,x,y);
    viterbiRegion = viterbiRegion | IN;    
    viterRegion1 = find(viterbiRegion==1);
    
    xNow = ceil(finalPath(i, :)./rows);
    yNow = rem(finalPath(i, :), rows);
    INNOW = inpolygon(regionX,regionY,xNow,yNow);
    INNOW1 = find(INNOW==1);
    
    if ~isempty(find(viterRegion1==index(i),1)) ||...
            ~isempty(intersect(INNOW1, index(1:i-1)))

        finalPath(i, :) = [];
        index(i) = [];
        res(i, :) = [];
        
        sign = 1;
    end   

    if sign == 0
        i = i+1;
    else
        sign = 0;
    end
end

x1 = res(:,1);
y1 = res(:,2);

final_num = length(x1);
fprintf('Final number of segmented nanodiscs is %d.\n', final_num);

%% Shrink the contours

finalPath=shrinkContours(finalPath, x1, y1, 0.20);


%% Draw the result
I1 = originalImage;
I2 = originalImage;
I3 = originalImage;

% draw forbidden lines
r1 = forbidHeiS:1:forbidHeiE;
c1 = ones(1,length(r1)).*forbidWidS;
c2 = forbidWidS:cols;
r2 = ones(1,length(c2)).*forbidHeiE;
whiteLine = sub2ind([rows,cols], [r1, r2], [c1, c2]);

r1 = ones(1,forbidWidE).*forbidHeiS;
c1 = 1:forbidWidE;
r2 = forbidHeiS:1:rows;
c2 = ones(1,length(r2)).*forbidWidE;
redLine = sub2ind([rows,cols], [r1, r2], [c1, c2]);

I1(whiteLine) = 255;
I2(whiteLine) = 255;
I3(whiteLine) = 255;
I1(redLine) = 255;
I2(redLine) = 0;
I3(redLine) = 0;

% draw nanodisc segmentations
regionX = repmat(1:cols,rows,1);
regionY = repmat((1:rows)', 1, cols);

w = 0.3;  % control the color depth
fL = length(x1);
for i = 1:fL
    % draw nanodisc centres
    cx1 = x1(i)-2:x1(i)+2;
    cx1(cx1<1) = 1;
    cx1(cx1>cols) = cols;
    cy1 = repmat(y1(i), 1, length(cx1));
       
    cy2 = y1(i)-2:y1(i)+2;
    cy2(cy2<1) = 1;
    cy2(cy2>rows) = rows;
    cx2 = repmat(x1(i), 1, length(cy2));
    nanoCentres = sub2ind([rows, cols], [cy1,cy2], [cx1,cx2]);
    I1(nanoCentres) = 0;
    I2(nanoCentres) = 0;
    I3(nanoCentres) = 255;

    % draw nanodisc contours
    [y, x] = ind2sub([rows, cols], finalPath(i, :));
    IN = inpolygon(regionX,regionY,x,y);
    boundaries = bwboundaries(IN);
    nanoRegion = sub2ind([rows,cols],boundaries{1}(:,1),boundaries{1}(:,2));
    
    I1(nanoRegion) = 255;
    I2(nanoRegion) = 0;
    I3(nanoRegion) = 0;
  
end 

% save final result
drawFrames = zeros(rows, cols, 3);
drawFrames(:,:,1) = I1;
drawFrames(:,:,2) = I2;
drawFrames(:,:,3) = I3;
drawFrames = uint8(drawFrames);
%imwrite(drawFrames, 'result.png');
imshow(drawFrames)
