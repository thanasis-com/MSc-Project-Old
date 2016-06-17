import theano
import theano.tensor as T
import numpy
from PIL import Image

from os import listdir
from os.path import isfile, join

import time

import numpy as np
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import lasagne
from lasagne.layers import InputLayer, Conv2DLayer, DenseLayer, MaxPool2DLayer, InverseLayer


#imports all image names from a specific path and turns them into numpy arrays,
#so that they can be fed to theano models

def loadImages(path, imageHeight, imageWidth, imageChannels):
	#get all filenames
	filenames = [f for f in listdir(path) if isfile(join(path, f))]

	print('Loading images...')
	start_time = time.time()

	#define the size of the image
	ImageSize = (imageHeight, imageWidth)
	#define the number of channels of the image
	NChannelsPerImage = imageChannels
	
	#the code below loads all the images
	imagesData = [ Image.open(path + '/' + f, 'r').getdata() for f in filenames ]
	
	#extracts the id of the image
	imageIDs=list()
	for f in filenames:
		id=int(filter(str.isdigit, f))
		imageIDs.append(id)

	imageIDs=numpy.array(imageIDs)	

	for i in imagesData :
	    assert i.size == ImageSize
	    assert i.bands == NChannelsPerImage
	 
	allImages = numpy.asarray(imagesData)
	nImages = len(filenames)
	if imageChannels==1:
		allImages = numpy.rollaxis(allImages, 1, 1).reshape(nImages, NChannelsPerImage, ImageSize[0], ImageSize[1])
	else:
		allImages = numpy.rollaxis(allImages, 2, 1).reshape(nImages, NChannelsPerImage, ImageSize[0], ImageSize[1])

	#sort the images according to their ID
	allImages=allImages[imageIDs.argsort()]

	end_time = time.time()
	print('Loaded and reshaped %d images in %.2f seconds' % (len(filenames), end_time-start_time))

	return allImages
	
	
#takes as input the output of the loadImages function
#returns the images with a single channel
def oneDimension(images):
	images=images[:,0:1,:,:]
	
	return images


def crop(images, cropPercentage):
	#get the original dimensions of the images
	originalXdim=images.shape[2]
	originalYdim=images.shape[3]

	#compute the new dimensions based on the cropPercentage
	newXdim=originalXdim*cropPercentage/100
	newYdim=originalYdim*cropPercentage/100

	#compute how many pixels from each side should be cropped 
	cropx=(originalXdim-newXdim)/2
	cropy=(originalYdim-newYdim)/2
	
	#crop images
	croppedImages=images[:, :, cropx:originalXdim-cropx-1, cropy:originalYdim-cropy-1]
	
	#get actual new image dimensions
	actualNewXdim=croppedImages.shape[2]
	actualNewYdim=croppedImages.shape[3]

	print('Cropped images from %d x %d to %d x %d' % (originalXdim, originalYdim, actualNewXdim, actualNewYdim))

	return croppedImages
 














