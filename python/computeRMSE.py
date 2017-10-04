import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from PIL import Image
import numpy
import numpy as np
import sys
import myTools
import scipy
from os import listdir
from os.path import isfile, join
from sklearn.metrics import mean_squared_error
from math import sqrt

#get the produced masks' path
imagePath=sys.argv[1]

#get all images' names
filenames = [f for f in listdir(imagePath) if isfile(join(imagePath, f))]

rmseList=list()

for f in filenames:
    #open the image
    myMask=Image.open(imagePath + '/' + f, 'r')
    #image as numpy array
    myMask=numpy.asarray(myMask)
    myMask=myTools.cropCenter1(myMask, 100)

    #open the expert mask
    theMask=Image.open('../../masksExpertTest/' + '/' + f, 'r')
    #image as numpy array
    theMask=numpy.asarray(theMask)

    thisRMSE = sqrt(mean_squared_error(theMask, myMask))

    rmseList.append(thisRMSE)



print(numpy.mean(rmseList))
