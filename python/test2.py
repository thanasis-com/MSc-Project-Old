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



dataSet=myTools.loadImages('../../images', 1024, 1024, 4)

dataSet=myTools.oneDimension(dataSet)

dataSet= dataSet.astype(numpy.uint8)

image=dataSet[0][0]


plt.show(plt.imshow(myTools.imgTransform(image), cmap=cm.binary))






