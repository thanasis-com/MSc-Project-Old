import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

from skimage import data, img_as_float, img_as_ubyte
from skimage import exposure
from skimage.transform import rotate
import pylab
import numpy
import theano
import theano.tensor as T
import scipy
import myTools
import pylab
import sys
from PIL import Image
import png
import math
from scipy import ndimage


masks=myTools.loadImages('../../masks', 819, 819, 1)


masks[masks>0]=1


for x in numpy.nditer(masks, op_flags=['readwrite']):
     if x==1:
             x[...]=0
     else:
	     x[...]=1

masks[masks==1]=0
masks[masks!=0]=1


masks=masks.astype(numpy.float32)

#dt=ndimage.distance_transform_edt(masks)

for i in xrange(masks.shape[0]):
	masks[i][0]=ndimage.distance_transform_edt(masks[i][0])


#for x in numpy.nditer(masks, op_flags=['readwrite']):
#     if x>20:
#             x[...]=20

masks[masks>20]=20

print(numpy.amax(masks))

masks=masks/numpy.amax(masks)

print(masks[0][0])

plt.show(plt.imshow(masks[0][0], cmap=cm.binary))



