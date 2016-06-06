function contours = fixContours(allPolygon)

%preallocating memory for the contours
contours=cell(length(allPolygon),1);

for i=1:length(allPolygon)
    %take each polygon
    polygon=allPolygon{i};
    
    polygonSize=length(polygon);
    
    j=1;
    
    %eliminates polygons annotated by mistake (one-point polygons)
    if(polygonSize<=2)
        continue;
    end
    
    %for each point of the polygon, check if its euclidean distance from
    %the next point is >1 and if yes, then add to the polygon all the
    %inbetween points
    while(j<=polygonSize)
        pointx=polygon(j,1);
        pointy=polygon(j,2);
        point=[pointx pointy];
        %taking care of the case we are trying to access the next to the
        %last point of the polygon
        try
            nextPointx=polygon(j+1,1);
            nextPointy=polygon(j+1,2);
        catch
            nextPointx=polygon(1,1);
            nextPointy=polygon(1,2);
        end
        nextPoint=[nextPointx nextPointy];
        
        %euclidean distance squared
        if(norm(nextPoint-point)^2>2)
            %compute how many inbetween points to create
            %+15 has been chosen by trial and error. it gives a good result
            numOfInbetween=abs(max((nextPoint(2)-point(2)+15),nextPoint(1)-point(1)+15));
            %create the inbetween points
            additionalPoints=[round(linspace(point(1),nextPoint(1),numOfInbetween));...
                round(linspace(point(2),nextPoint(2),numOfInbetween))];
            %transposing to get the format I want
            additionalPoints=transpose(additionalPoints);
            
            %taking care of the case we are trying to concatenate the next to the
            %last point of the polygon
            try
                polygon=vertcat(polygon(1:j-1,:), additionalPoints, polygon(j+1:end,:));
            catch
                polygon=vertcat(polygon(1:j-1,:), additionalPoints);
            end
            
            polygonSize=length(polygon);
            j=j+numOfInbetween;
        else
            %if the two points are neighbours, then there is no need to add
            %inbetween points
            j=j+1;
        end
    end
    
    contours{i}=polygon;

end
