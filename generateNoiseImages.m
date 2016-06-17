
%get all images
images=dir('DataSet/*.png');

for j=1:length(images)
    
     %read a new image
    I=imread(cat(2, 'DataSet/', images(j).name));
    I = rgb2gray(I);
    [rows,cols]=size(I);
    
    %extract image id (exrtacted by the name of the file)
    imageID=strtok(cat(2, images(j).name), 'p');
    
    %add noise to the image
    noisyImage=imnoise(I,'gaussian', 0, 0.1);
    
    %save each noisy image to a file
    imwrite(noisyImage, cat(2, 'noiseImages/noise', imageID, '.png'));
end
