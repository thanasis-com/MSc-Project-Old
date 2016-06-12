import myTools

import theano
import numpy
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


dataSet=myTools.loadImages('/home/athanasiostsiaras/Downloads/images', 1024, 1024, 4)

dataSet=myTools.oneDimension(dataSet)
	
#plt.show(plt.imshow(dataSet[0][0], cmap=cm.binary))

masks=myTools.loadImages('/home/athanasiostsiaras/Downloads/masks', 819, 819, 1)

#plt.show(plt.imshow(masks[0][0], cmap=cm.binary))



