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
image=myTools.cropCenter1(image, 83.1)
#split image to 4
splits=myTools.augmentData(image.reshape(1,1,image.shape[0],image.shape[1]), numOfTiles=4, overlap=False, imageWidth=image.shape[0], imageHeight=image.shape[1])
#another vital type casting
splits=splits.astype(numpy.float32)
#keep only the 4 original tiles
splits=splits[0:4,:,:,:]
#setting parameters for the network
data_size=(None,1,splits[0][0].shape[0],splits[0][0].shape[1])
#load the pretrained network
myNet=myTools.createPretrainedNN(data_size)
#make predictions for the 4 tiles
print(splits.dtype)
res=myNet(splits)
#crop the center of the predictions
res=myTools.cropCenter(res, 93)
#concatenate on the x axis
top=np.concatenate((res[0][0],res[2][0]),axis=1)
bot=np.concatenate((res[1][0],res[3][0]),axis=1)
#concatenate the two halves to get the full image
res=np.concatenate((top,bot),axis=0)

plt.show(plt.imshow(res, cmap=cm.binary))
scipy.misc.imsave(outName, res)
