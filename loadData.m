function imdb=loadData(rawDirectory, truthDirectory)

imdb.images.id=[];
imdb.images.data=[];
imdb.images.label=[];
imdb.images.set=[];

%open the file that contains the images
files=dir(cat(2, rawDirectory,'*.jpg'));
id=1;
for file=files'
    %add a new id
    imdb.images.id=cat(2, imdb.images.id, id);
    newData=rgb2gray(im2single(imread(cat(2, rawDirectory, file.name)))) ;
    %add the new image matrix
    imdb.images.data=cat(3, imdb.images.data, newData);
    newLabelImage=rgb2gray(im2single(imread(cat(2, truthDirectory, file.name)))) ;
    neg=(newLabelImage>=0.4);
    pos=(newLabelImage<0.2);
    newLabel = zeros(size(pos),'single') ;
    newLabel(pos) = +1 ;
    newLabel(neg) = -1 ;
    %add the new label matrix
    imdb.images.label=cat(3, imdb.images.label, newLabel);
    numOfFiles=size(files);
    numOfFiles=numOfFiles(:,1);
    if id<numOfFiles*0.8
        imdb.images.set=cat(2, imdb.images.set, 1);
    else
        imdb.images.set=cat(2, imdb.images.set, 2);
    end
    id=id+1;
end