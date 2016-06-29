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



#dataSet=myTools.loadImages('../../images', 1024, 1024, 4)

#dataSet=myTools.oneDimension(dataSet)

#dataSet= dataSet.astype(numpy.uint8)

#image=dataSet[0][0]

masks=myTools.loadImages('../../masks', 819, 819, 1)


for x in numpy.nditer(masks, op_flags=['readwrite']):
     if x>50:
             x[...]=1
     else:
	     x[...]=0

masks=masks.astype(numpy.float32)


temp=myTools.augmentData(masks, numOfTiles=4, overlap=False, imageWidth=819, imageHeight=819)

plt.show(plt.imshow(temp[0], cmap=cm.binary))






