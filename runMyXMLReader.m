

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

%saveas(gcf,cat(2, 'annotation', '1', '.png'));

hold off;
