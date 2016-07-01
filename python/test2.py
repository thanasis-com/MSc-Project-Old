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

import sys


masks=myTools.loadImages('../../masks', 819, 819, 1)

temp=0
whole=0

for x in numpy.nditer(masks, op_flags=['readwrite']):
     whole+=1
     if x>0:
             x[...]=1
	     temp+=1
     else:
 	     x[...]=0

print(masks[0][0])
plt.show(plt.imshow(masks[0][0]))
print(temp)
print(whole)
print(float(temp)/whole)

