import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

from skimage import data, img_as_float, img_as_ubyte
from skimage import exposure
from skimage.transform import rotate

import numpy
import theano
import theano.tensor as T

import myTools


def myCostFunction(a, b):
  

    r=np.float32(0.01)

    sensitivity=r*numpy.sum(((b - a)**2)*b)/numpy.sum(b)
    
    specificity=(1-r)*numpy.sum(((b - a)**2)*(1-b))/numpy.sum(1-b)
	
    return sensitivity+specificity



### MASKS
masks=myTools.loadImages('../../masks', 819, 819, 1)

for x in numpy.nditer(masks, op_flags=['readwrite']):
     if x>0:
             x[...]=1

masks=masks.astype(numpy.float32)

masks=myTools.augmentData(masks, numOfTiles=4, overlap=False, imageWidth=819, imageHeight=819)

masks=masks.astype(numpy.float32)

print(myCostFunction(masks,masks))


