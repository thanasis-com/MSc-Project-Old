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
	images = [ Image.open(path + '/' + f, 'r') for f in filenames ]
	for i in images :
	    assert i.size == ImageSize
	    assert len(i.getbands()) == NChannelsPerImage
	 
	ImageShape =  (1,) + ImageSize + (NChannelsPerImage,)
	allImages = [ numpy.fromstring(i.tostring(), dtype='uint8', count=-1, sep='') for i in images ]
	allImages = [ numpy.rollaxis(a.reshape(ImageShape), 3, 1) for a in allImages ]

	return allImages


def dumpA(images):
	fixedImages=list()

	for i in range(0, len(images)-1):
		fixedImages.append(images[i][:,0:3,:,:])

	return fixedImages
