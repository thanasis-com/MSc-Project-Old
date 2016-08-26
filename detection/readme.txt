Here is an introduction for how to run this code:

1. The parameters are gathered in the 'setParameters.m' file. You can set your own parameters and specify your own testing image in this file.

2. Please use the commands 'mex projectionMap.cpp' and 'mex keypoints.cpp' to build the mex files before running the code.

3.1 If the testing image is the image with gold markers, please run 'nanodiscAndGold.m' file, the result will be stored in the 'result.png' image. 

3.2 If the testing image is the image without gold markers, please run 'nanodiscNoGold.m' file, the result will be stored in the 'result.png' image.
    

Note: 
1. The parameters in the 'setParameters.m' file are not finely tuned, so please tune the parameters according to the images which you want to segment in order to get your ideal result.
2. I slightly changed some codes in my previous project, so now the result possibly will not be the same with the result in my nanodisc detection and segmentation paper. 


Other information:
All the images (with and without gold markers) are in the 'images' folder.
The ground truth annotations from two experts are in the 'groundTruth' folder.
The ground truth for gold markers is in the 'goldMarkers' folder.

I wrote a function 'NewXMLReader.m' to extract their annotation information from the xml files. You can run 'runXMLReader.m' to see how it works. 