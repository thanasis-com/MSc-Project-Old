
I=imread('DataSet\35png.png');
I = rgb2gray(I);
[rows,cols]=size(I);

xmlFile='DataSet\35anno.xml';

% forbidden lines
r = 0.8;
forbidWid = round(r*cols);
forbidHei = round(r*rows);
forbiddenLines.forbidWidS = floor((cols-forbidWid)/2)+1;
forbiddenLines.forbidWidE = forbiddenLines.forbidWidS+forbidWid-1;
forbiddenLines.forbidHeiS = floor((rows-forbidHei)/2)+1;
forbiddenLines.forbidHeiE = forbiddenLines.forbidHeiS+forbidHei-1;

[N, grTru, allPolygon, radius, polyCen, polyArea] = myXMLReader(rows, cols, I, xmlFile, forbiddenLines);

%fix possible gaps in the contours
contours=fixContours(allPolygon);

%initialise the mask
mask=zeros(rows,cols);

for i=1:length(contours)
    for j=1:length(contours{i})
        x=contours{i}(j,1);
        y=contours{i}(j,2);
        mask(x,y)=1;
    end
end

%create filter for bluring the mask
myFilter=fspecial('gaussian', [6 6], 1.5);

%blur the image
bluredMask=imfilter(mask,myFilter);

%crop the image to the desired size
bluredMask=bluredMask(forbiddenLines.forbidWidS:forbiddenLines.forbidWidE, forbiddenLines.forbidHeiS:forbiddenLines.forbidHeiE);

imshow(bluredMask);

