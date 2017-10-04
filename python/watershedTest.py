import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from PIL import Image
import numpy
import numpy as np
import sys
import myTools
import scipy

#get the image path
imagePath=sys.argv[1]
#open the image
image=Image.open(imagePath)
#image as numpy array
image=numpy.asarray(image)

from skimage.morphology import watershed
from skimage.feature import peak_local_max
import skimage
from scipy import ndimage
markers[~image] = -1
labels_rw = segmentation.random_walker(image, markers)

plt.show(plt.imshow(labels_rw, cmap=cm.binary))
