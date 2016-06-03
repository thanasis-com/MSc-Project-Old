function [N, grTru, allPolygon, radii, polyCen, polyArea] = myXMLReader( heightSize, widthSize, I, xmlFile, forbiddenLines )
%Extract nanodisc information from the ground truth annotations. 
%
% Parameters:
%    heightSize: height of the image
%    widthSize: width of the image
%    I: image matrix
%    xmlFile: location of the xml file 
%    forbiddenLines: region of forbidden lines   
%
% Returns:
%    N: number of nanodisc annotations
%    grTru: the indices of points in each annotation region
%    allPolygon: the indices of annotation points for each nanodisc
%    radii: radius of each nanodisc annotation
%    polyCen: centre of each nanodisc annotation
%    polyArea: area of each nanodisc annotation
%


xmlDoc = xmlread(xmlFile);   % read xml file 

% forbidden lines
forbidWidS = forbiddenLines.forbidWidS;
forbidWidE = forbiddenLines.forbidWidE;
forbidHeiS = forbiddenLines.forbidHeiS;
forbidHeiE = forbiddenLines.forbidHeiE;
 


%% extract the polygons  
polyArray1 = xmlDoc.getElementsByTagName('coords');  % put 'coords' nodes into 'polyArray1'
polyArray2 = xmlDoc.getElementsByTagName('label');

numberPolygons = 0;
N = polyArray1.getLength;
allPolygon = cell(N,1);
preAllpoints = [];

% calculate the number of annotations and the coordinates of points in each annotation      
for i = 0 : N-1
    
    % judge whether the current annotation has been eliminated
    nodeContent1 = char(polyArray2.item(i).getFirstChild.getData);    % extract the information in current node
    labelDate1 = str2double(nodeContent1);
    if i < polyArray1.getLength-1
        nodeContent2 = char(polyArray2.item(i+1).getFirstChild.getData);    % extract the label in next node
        labelDate2 = str2double(nodeContent2);
    end

    if labelDate1 == -1 || labelDate2 == -1
        continue;
    end  
    
    allPoints = [];
    thisItem = polyArray1.item(i);  
    childNode = thisItem.getFirstChild ;  
    while ~isempty(childNode)  % go through all the child nodes
        
        childchild = childNode.getFirstChild ; 
        if childchild.getNodeType == childchild.ELEMENT_NODE ;    % check if current node has child nodes   
            childchildNm = char(childchild.getTagName);        % get the name of the current node 
            childchildX = char(childchild.getFirstChild.getData);    % get the data in current node
            childchildX = str2double(childchildX);
            x = -childchildX;
        end  
        childchild = childchild.getNextSibling;     % go to next node  
        
        if childchild.getNodeType == childchild.ELEMENT_NODE ;    % check if current node has child nodes
            childchildNm = char(childchild.getTagName);        % get the name of the current node  
            childchildY = char(childchild.getFirstChild.getData);    % get the data in current node
            childchildY = str2double(childchildY);
            y = -childchildY;
        end  
               
        allPoints = [allPoints; [x, y]];                     
        childNode = childNode.getNextSibling;     % go to next node 
    end  
    
    if isequal(allPoints, preAllpoints) 
        continue;
    end
    preAllpoints = allPoints;
        
    
    boundx = allPoints(:, 1);
    boundy = allPoints(:, 2);   
    
    %eliminate all the vesicles that are entirely outside the forbidden
    %lines
    if all(boundx>forbidWidE) || all(boundy<forbidHeiS) ...
            || all(boundx<forbidWidS) || all(boundy>forbidHeiE)
        continue;
    end
    
    %eliminate the parts of the vesicles that are outside the right forbidden
    %line
    unwantedPoints=find(boundx>forbidWidE);
    if ~isempty(unwantedPoints)
        tempIndex = true(size(boundx, 1), 1);
        tempIndex(unwantedPoints) = false;
        boundx=boundx(tempIndex, :);
        boundy=boundy(tempIndex, :);
        allPoints=horzcat(boundx,boundy);
    end
    
    %eliminate the parts of the vesicles that are outside the left forbidden
    %line
    unwantedPoints=find(boundx<forbidWidS);
    if ~isempty(unwantedPoints)
        tempIndex = true(size(boundx, 1), 1);
        tempIndex(unwantedPoints) = false;
        boundx=boundx(tempIndex, :);
        boundy=boundy(tempIndex, :);
        allPoints=horzcat(boundx,boundy);
    end
    
    %eliminate the parts of the vesicles that are outside the bottom forbidden
    %line
    unwantedPoints=find(boundy<forbidHeiS);
    if ~isempty(unwantedPoints)
        tempIndex = true(size(boundx, 1), 1);
        tempIndex(unwantedPoints) = false;
        boundx=boundx(tempIndex, :);
        boundy=boundy(tempIndex, :);
        allPoints=horzcat(boundx,boundy);
    end
        
    %eliminate the parts of the vesicles that are outside the top forbidden
    %line
    unwantedPoints=find(boundy>forbidHeiE);
    if ~isempty(unwantedPoints)
        tempIndex = true(size(boundx, 1), 1);
        tempIndex(unwantedPoints) = false;
        boundx=boundx(tempIndex, :);
        boundy=boundy(tempIndex, :);
        allPoints=horzcat(boundx,boundy);
    end
    
    if(isempty(allPoints))
        continue;
    end
    
    numberPolygons = numberPolygons+1;
    allPolygon{numberPolygons,1} = allPoints;        
    
end  

allPolygon(numberPolygons+1:N) = [];
N = numberPolygons;

% % compute the areas of annotations and their radii
% regionX = zeros(heightSize, widthSize);
% for i = 1:heightSize
%     regionX(i,:) = i;
% end
% regionY = zeros(heightSize, widthSize);
% for i = 1:widthSize
%     regionY(:,i) = i;
% end
% 
 polyCen = zeros(N,2);  % annotation centre
 polyArea = zeros(N,1);  % annotation area
% finalRegion = zeros(heightSize, widthSize);
 grTru = cell(N, 1);
 radii = zeros(N,1);
% for i = 1:N
%     xv = allPolygon{i,1}(:,2);
%     yv = allPolygon{i,1}(:,1);        
%     IN = inpolygon(regionX,regionY,xv,yv);
%     grTru{i,1} = find(IN == 1);
%     numPoints = length(grTru{i,1});
%     radii(i) = round(sqrt(numPoints/pi));
%     finalRegion = finalRegion | IN;
%     
%     % calculate the annotation area and centre coordinates
%     A = numPoints; 
%     %centroids = regionprops(IN,'centroid');    
%     %polyCen(i, :) = centroids.Centroid;
%     polyArea(i) = A;
%     
% end



