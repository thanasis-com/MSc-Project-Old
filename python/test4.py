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
#set output filename
outName=sys.argv[2]
#open the image
image=Image.open(imagePath)
#image as numpy array
image=numpy.asarray(image)
#keep only one dimension
image=image[:,:,0]
#most needed type casting
image=image.astype(numpy.uint8)
#crop the center
image=myTools.cropCenter1(image, 80)
image=image.astype(numpy.float32)

plt.show(plt.imshow(image, cmap=cm.binary))
scipy.misc.imsave(outName, image)
