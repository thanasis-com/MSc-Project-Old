
%get all images
images=dir('DataSet/*.png');
%get all xmls files
xmls=dir('DataSet/*.xml');

numOfVesicles=0;
for j=1:length(images)
    
    %read a new image
    I=imread(cat(2, 'DataSet/', images(j).name));
    I = rgb2gray(I);
    [rows,cols]=size(I);
    
    %extract image id (exrtacted by the name of the file)
    imageID=strtok(cat(2, images(j).name), 'p');
    
    
    %read the respective xml file
    xmlFile=cat(2, 'DataSet/', xmls(j).name);
    
    % forbidden lines
    r = 0.8;
    forbidWid = round(r*cols);
    forbidHei = round(r*rows);
    forbiddenLines.forbidWidS = floor((cols-forbidWid)/2)+1;
    forbiddenLines.forbidWidE = forbiddenLines.forbidWidS+forbidWid-1;
    forbiddenLines.forbidHeiS = floor((rows-forbidHei)/2)+1;
    forbiddenLines.forbidHeiE = forbiddenLines.forbidHeiS+forbidHei-1;

    [N, grTru, allPolygon, radius, polyCen, polyArea] = myXMLReader(rows, cols, I, xmlFile, forbiddenLines);
    
    numOfVesicles=numOfVesicles+N;
    
    %% plot the ground truth
    figure;
    imshow(I);
    hold on;
    axis equal;
    axis([0 1024 0 1024]);

    plot([1,forbiddenLines.forbidWidE,forbiddenLines.forbidWidE,cols], [forbiddenLines.forbidHeiS,forbiddenLines.forbidHeiS,forbiddenLines.forbidHeiE,forbiddenLines.forbidHeiE], 'r', 'linewidth',2);
    hold on;
    plot([forbiddenLines.forbidWidS,forbiddenLines.forbidWidS,forbiddenLines.forbidWidE], [forbiddenLines.forbidHeiS,forbiddenLines.forbidHeiE,forbiddenLines.forbidHeiE], 'r','linewidth',2);
    hold on;

    for i=1:N
        xp = [allPolygon{i,1}(:,1); allPolygon{i,1}(1,1)];
        yp = [allPolygon{i,1}(:,2); allPolygon{i,1}(1,2)];
        plot(xp, yp, 'b', 'linewidth',1);
        hold on;
    end
    
    %save each image to a file
    saveas(gcf,cat(2, 'annotatedImages/annotation', imageID, '.png'));
    hold off;
end

disp(cat(2, num2str(numOfVesicles), ' vesicles in total'))

