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
import scipy
import myTools

import sys
from PIL import Image

from scipy import misc

face = misc.imread('output1.png')
type(face)      

print(face.shape) 
print(face.dtype)

print(face)

print(numpy.amax(face))
print(numpy.amin(face))
print(numpy.mean(face))

plt.show(plt.imshow(face, cmap=cm.binary))
