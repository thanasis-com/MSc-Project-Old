import theano
import numpy
from PIL import Image

from os import listdir
from os.path import isfile, join


#imports all image names from a specific path and turns them into numpy arrays,
#so that they can be fed to theano models

def loadImages(path, imageHeight, imageWidth, imageChannels):
	#get all filenames
	filenames = [f for f in listdir(path) if isfile(join(path, f))]
	#define the size of the image
	ImageSize = (imageHeight, imageWidth)
	#define the number of channels of the image
	NChannelsPerImage = imageChannels
	
	#the code below is a fast way to load all the images
	imagesData = [ Image.open(path + '/' + f, 'r').getdata() for f in filenames ]
	for i in imagesData :
	    assert i.size == ImageSize
	    assert i.bands == NChannelsPerImage
	 
	allImages = numpy.asarray(imagesData)
	nImages = len(filenames)
	allImages = numpy.rollaxis(allImages, 2, 1).reshape(nImages, NChannelsPerImage, ImageSize[0], ImageSize[1])

	return allImages
	
def dumpAA(images):
	images=images[:,0:3,:,:]
	
	return images

#takes as input the output of the loadImages function
#returns the RGBA array as a RGB array by dumping the A dimension

def dumpA(images):

	fixedImages=list()

	for i in range(0, len(images)-1):
		fixedImages.append(images[i][:,0:3,:,:])

	return fixedImages
