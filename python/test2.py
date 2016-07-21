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

one = np.load("outfile1.npy")
two = np.load("outfile2.npy")


plt.show(plt.imshow(one, cmap=cm.binary))
#pylab.savefig('out7.png', bbox_inches='tight')
plt.show(plt.imshow(two, cmap=cm.binary))
#pylab.savefig('out8.png', bbox_inches='tight')





