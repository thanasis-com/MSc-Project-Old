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

import sys
from PIL import Image
import png

test=numpy.array([[0.58683, 0.3219283], [0.56384, 0.48238]])

#png.Writer.write(pngfile, numpy.reshape(test, (-1, column_count*plane_count)))

numpy.save("outfile.npy", test)


