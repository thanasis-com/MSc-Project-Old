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
#image=myTools.cropCenter1(image, 100)
#this step is mysteriously needed
image=myTools.augmentMasks(image.reshape(1,1,image.shape[0],image.shape[1]), numOfTiles=1, overlap=False, imageWidth=image.shape[0], imageHeight=image.shape[1])
#another vital type casting
image=image.astype(numpy.float32)
#setting parameters for the network
data_size=(None,1,image[0][0].shape[0],image[0][0].shape[1])

#load the pretrained network
myNet1=myTools.createPretrainedNN2(data_size, modelFile='model020New.npz', filters=64)
#make predictions for the image
res1=myNet1(image)
#crop the center of the mask
res1=myTools.cropCenter(res1, 80)

#load the pretrained network
myNet2=myTools.createPretrainedNN2(data_size, modelFile='model20NewsallLR688images.npz', filters=64)
#make predictions for the image
res2=myNet2(image)
#crop the center of the mask
res2=myTools.cropCenter(res2, 80)

#load the pretrained network
myNet3=myTools.createPretrainedNN2(data_size, modelFile='model20NewSmallLR.npz', filters=64)
#make predictions for the image
res3=myNet3(image)
#crop the center of the mask
res3=myTools.cropCenter(res3, 80)

#load the pretrained network
myNet4=myTools.createPretrainedNN2(data_size, modelFile='model030New.npz', filters=64)
#make predictions for the image
res4=myNet4(image)
#crop the center of the mask
res4=myTools.cropCenter(res4, 80)

#load the pretrained network
myNet5=myTools.createPretrainedNN2(data_size, modelFile='model080-500epochs.npz', filters=64)
#make predictions for the image
res5=myNet5(image)
#crop the center of the mask
res5=myTools.cropCenter(res5, 80)

#load the pretrained network
myNet6=myTools.createPretrainedNN2(data_size, modelFile='model080New.npz', filters=64)
#make predictions for the image
#res6=myNet6(image)
#crop the center of the mask
#res6=myTools.cropCenter(res6, 80)

#load the pretrained network
myNet7=myTools.createPretrainedNN2(data_size, modelFile='modelNew50.npz', filters=50)
#make predictions for the image
res7=myNet7(image)
#crop the center of the mask
res7=myTools.cropCenter(res7, 80)

#load the pretrained network
myNet8=myTools.createPretrainedNN2(data_size, modelFile='modelbest.npz', filters=35)
#make predictions for the image
res8=myNet8(image)
#crop the center of the mask
res8=myTools.cropCenter(res8, 80)


print(numpy.mean(res1))
print(numpy.mean(res2))
print(numpy.mean(res3))
print(numpy.mean(res4))
print(numpy.mean(res5))
#print(numpy.mean(res6))
print(numpy.mean(res7))
print(numpy.mean(res8))

# Create a numpy array of floats to store the average (assume RGB images)
res=numpy.zeros((res1[0][0].shape[0], res1[0][0].shape[1]),numpy.float)

imlist=[res1, res2, res3, res4, res5, res7, res8]
# Build up average pixel intensities, casting each image as an array of floats
for im in imlist:
    newRes=im
    res=res+newRes/len(imlist)

plt.show(plt.imshow(res[0][0], cmap=cm.binary))
scipy.misc.imsave(outName, res[0][0])
