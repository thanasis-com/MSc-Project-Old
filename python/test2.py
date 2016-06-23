import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

from skimage import data, img_as_float
from skimage import exposure

import numpy
import theano
import theano.tensor as T

import myTools


# Load an example image
img = myTools.loadImages('/home/athanasiostsiaras/Downloads/images', 1024, 1024, 4)
img= myTools.oneDimension(img)
img=img.astype(theano.config.floatX)

imgage=img[0][0].crop(0,0,100,100)

plt.show(imshow(imgage))













