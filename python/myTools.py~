import theano
import numpy
from PIL import Image

from os import listdir
from os.path import isfile, join

import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#imports all image names from a specific path and turns them into numpy arrays,
#so that they can be fed to theano models

def loadImages(path, imageHeight, imageWidth, imageChannels):
	#get all filenames
	filenames = [f for f in listdir(path) if isfile(join(path, f))]
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

	return allImages
	
	
#takes as input the output of the loadImages function
#returns the images with a single channel
def oneDimension(images):
	images=images[:,0:1,:,:]
	
	return images

 

def myConvNet(X_train, y_train):
	
	myNet = NeuralNet(
	    layers=[('input', layers.InputLayer),
		    ('conv1', layers.Conv2DLayer),
		    ('maxpool1', layers.MaxPool2DLayer),
		    ('conv2', layers.Conv2DLayer),
		    ('output', layers.MaxPool2DLayer),
		    ],
	    # input layer
	    input_shape=(None, 1, 1024, 1024),
	    # layer conv1
	    conv1_num_filters=10,
	    conv1_filter_size=(5, 5),
	    conv1_nonlinearity=lasagne.nonlinearities.rectify,
	    conv1_W=lasagne.init.GlorotUniform(),  
	    # layer maxpool1
	    maxpool1_pool_size=(2, 2),    
	    # layer output
	    conv2d2_num_filters=10,
	    conv2d2_filter_size=(5, 5),
	    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
	   
	    # optimization method params
	    update=nesterov_momentum,
	    update_learning_rate=theano.shared(float32(0.01)),
	    update_momentum=theano.shared(float32(0.9)),
	    max_epochs=5,
	    verbose=1,
	    )
	# Train the network
	nn = myNet.fit(X_train, y_train)

















