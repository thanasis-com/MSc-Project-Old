function [] = detectLipos(filename)
%Detects lipoprotein particles in an image
%
%Patameters:
%           filename: the name of the image to be processed
commandStr=['python createMasksBig.py ../../images/',filename,' segMask.png'];

[status,result]=system(commandStr);

detectNanodiscs(filename, 'segMask.png')

