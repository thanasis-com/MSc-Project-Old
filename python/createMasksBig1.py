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
image=myTools.cropCenter1(image, 100)
#this step is mysteriously needed
image=myTools.augmentMasks(image.reshape(1,1,image.shape[0],image.shape[1]), numOfTiles=1, overlap=False, imageWidth=image.shape[0], imageHeight=image.shape[1])
#another vital type casting
image=image.astype(numpy.float32)
#setting parameters for the network
data_size=(None,1,image[0][0].shape[0],image[0][0].shape[1])
#load the pretrained network
myNet=myTools.createPretrainedNN(data_size)
#make predictions for the image
res=myNet(image)
#crop the center of the mask
res=myTools.cropCenter(res, 80)


plt.show(plt.imshow(res[0][0], cmap=cm.binary))
scipy.misc.imsave(outName, res[0][0])
