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


argLR=float(sys.argv[1])

argWD=float(sys.argv[2])

print('Learning rate: %f' % (argLR))
print('Weight decay: %f' % (argWD))
