function [shrinkedPath] = shrinkContours(finalPath, centerX, centerY, percentage)

[y, x]=ind2sub([1024, 1024], finalPath);

for i=1:length(centerX)
    numOfPoints=size(finalPath);
    numOfPoints=numOfPoints(2);
    for j=1:numOfPoints
 
        if(centerX(i)>x(i,j))
            x(i,j)=ceil(x(i,j)+(centerX(i)-x(i,j))*percentage);
        else
            x(i,j)=floor(x(i,j)-(x(i,j)-centerX(i))*percentage);
        end
        if(centerY(i)>y(i,j))
            y(i,j)=ceil(y(i,j)+(centerY(i)-y(i,j))*percentage);
        else
            y(i,j)=floor(y(i,j)-(y(i,j)-centerY(i))*percentage);
        end
        
    end
 
end

shrinkedPath=sub2ind([1024, 1024], y, x);

