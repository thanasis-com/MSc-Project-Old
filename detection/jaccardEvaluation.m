function [ jaccard_value, numRight ] = jaccardEvaluation( groundTruthPara, nanoArea, I, rows, cols, x1, y1, finalPath, forbiddenLines )
%Evaluate the segmentation performance 
%
% Parameters:
%    groundTruthPara: ground truth information from function 'NewXMLReader.m'
%    nanoArea: area for each segmentation
%    I: image matrix
%    rows: height of the image
%    cols: width of the image
%    x1: x coordinates of segmentation centres
%    y1: y coordinates of segmentation centres
%    finalPath: contours of all the segmentations
%    forbiddenLines: region of forbidden lines  
%
% Returns:
%    jaccard_value: jaccard index values for all the correct segmentations
%    numRight: number of correct segmentations
%


% ground truth parameters
N = groundTruthPara.N;
grTru = groundTruthPara.grTru;
allPolygon = groundTruthPara.allPolygon;
polyCen = groundTruthPara.polyCen;


% forbidden lines
forbidWidS = forbiddenLines.forbidWidS;
forbidWidE = forbiddenLines.forbidWidE;
forbidHeiS = forbiddenLines.forbidHeiS;
forbidHeiE = forbiddenLines.forbidHeiE;


figure;
imshow(I);
hold on;

% plot forbidden lines
plot([1,forbidWidE,forbidWidE,cols], [forbidHeiS,forbidHeiS,forbidHeiE,forbidHeiE], 'r', 'linewidth',3);
hold on;
plot([forbidWidS,forbidWidS,forbidWidE], [forbidHeiS,forbidHeiE,forbidHeiE], 'w','linewidth',3);
hold on;

% accuracy evaluation
index = (x1-1).*rows+y1;
copyIndex = index;
numRight = 0;   % Right segmentation number
correctPoints = [];

polyCen = abs(polyCen);


% the weight between Jaccard and area difference
areaJacW = 0.7;
for i=1:N    % for each nanodisc in the ground truth
    
    maxRes = 0;
    num_det = length(index);
    
    % find its corresponding segmentation 
    for j=1:num_det
        % if the centre of the segmentation is in the ground truth nanodisc,
        % we think it is a right segmentation. So we may have more than one
        % right segmentation.
        if ~isempty(find(grTru{i,1} == index(j), 1))
            
            jac = length(intersect(grTru{i,1},nanoArea{j,1})) /...
                length(union(grTru{i,1},nanoArea{j,1}));  % Jaccard index
            
            areaDiff = abs(length(grTru{i, 1}) - length(nanoArea{j,1}))*100/...
                length(grTru{i, 1});  % area difference
             
            absAreaDiff = abs(length(grTru{i, 1}) - length(nanoArea{j,1})); 
            
            % areaJacRp is used to find the best segmentation for the ground
            % truth nanodisc among its corresponding right segmentations.
            areaJacRp = areaJacW*(1/(areaDiff+eps)) + (1-areaJacW)*jac;
            
            % record the best corresponding segmentation
            if areaJacRp > maxRes
                maxRes = areaJacRp;
                maxPoint = [index(j), jac, areaDiff, absAreaDiff];
            end            
            
        end
    end
    
    
    if maxRes ~= 0
        numRight = numRight+1;
        
        % draw correct segmentations
        iPath = find(copyIndex == maxPoint(1));
        [indexy, indexx] = ind2sub([rows, cols], finalPath(iPath, :));
        plot([indexx, indexx(1)], [indexy,indexy(1)], 'b','linewidth',1);
        hold on;
        plot(x1(iPath), y1(iPath), 'b*', 'markersize', 6,'linewidth',1);
        hold on;
        
        correctPoints = [correctPoints; maxPoint];        
        
        nunDex = find(index == maxPoint(1));
        index(nunDex) = [];
        nanoArea(nunDex) = [];
        
        continue;
        
    end
     
    % draw missing segmentations
    plot(polyCen(i,1),polyCen(i,2),'gs', 'markersize', 6,'linewidth',1);
    hold on;
    xm = [allPolygon{i,1}(:,1); allPolygon{i,1}(1,1)];
    ym = [allPolygon{i,1}(:,2); allPolygon{i,1}(1,2)];
    plot(xm, ym, 'g-', 'linewidth', 1);
    hold on;    
    

end

% draw wrong segmentations
fL = length(index);
for i = 1:fL
    iPath = find(copyIndex==index(i));
    indexx = ceil(finalPath(iPath, :)./rows);
    indexy = rem(finalPath(iPath, :), rows);
    plot(x1(iPath), y1(iPath), 'r^', 'markersize', 6,'linewidth',1);
    hold on;

    plot([indexx, indexx(1)], [indexy,indexy(1)], 'r','linewidth',1);
    hold on;
end  

hold off;


jaccard_value = round(correctPoints(:,2)*100)/100;


end

