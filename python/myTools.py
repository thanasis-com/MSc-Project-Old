import theano
import theano.tensor as T
import numpy
from PIL import Image

from os import listdir
from os.path import isfile, join

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

#def crop(images, dimX, dimY):
	

 

def myConvNet(X_train, y_train):
	import numpy
	myNet = NeuralNet(
	        layers=[('input', layers.InputLayer),
		        ('conv1', layers.Conv2DLayer),
		        ('maxpool1', layers.MaxPool2DLayer),
		        ('unpool1', layers.InverseLayer),
		        ('output', layers.InverseLayer),
		       ],
	    	# input layer
	    	input_shape=(None, 1, 819, 819),
	    	# layer conv1
	    	conv1_num_filters=10,
	    	conv1_filter_size=(5, 5),
	    	conv1_nonlinearity=lasagne.nonlinearities.rectify,
	    	conv1_W=lasagne.init.GlorotUniform(),  
	    	# layer maxpool1
	    	maxpool1_pool_size=(2, 2),  
	    
	    	#layer unpool1
	    	unpool1_layer=['maxpool1'],
  
	    	# layer output
	    	output_layer=['conv1'],
	   
	    	# optimization method params
	    	update=nesterov_momentum,
	    	update_learning_rate=theano.shared(0.01),
	    	update_momentum=theano.shared(0.9),
	    	max_epochs=1,
	    	verbose=1,
	    	)
	# Train the network
	nn = myNet.fit(X_train, y_train)

def myConvNet2(X_train, y_train):

	l_in = InputLayer((None, 1, 819, 819), name="input_layer")
	l1 = Conv2DLayer(l_in, num_filters=10, filter_size=5, name="convolutional1")
	l2 = MaxPool2DLayer(l1, pool_size=2, name="maxpooling1")
	l_u2 = InverseLayer(l2, l2, name="unpooling1")
	l_u1 = InverseLayer(l_u2, l1, name="deconvolution1")

	net = NeuralNet(
    		l_u1,
    		update=nesterov_momentum,
    		update_learning_rate=0.01,
    		update_momentum=0.9,
    		max_epochs=2,
    		verbose=1,
    		)
	nn=net.fit(X_train, y_train)


def myLasagneNet(X_train, y_train):
	l_in = InputLayer((None, 1, 819, 819))
	l1 = Conv2DLayer(l_in, num_filters=10, filter_size=5)
	l2 = MaxPool2DLayer(l1, pool_size=2)
	l_u2 = InverseLayer(l2, l2)
	l_u1 = InverseLayer(l_u2, l1)

	input_var = T.tensor4('inputs')
	target_var = T.tensor4('targets')

	prediction = lasagne.layers.get_output(l_u1)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
	loss = loss.mean()
	params = lasagne.layers.get_all_params(l_u1, trainable=True)
	updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

	test_prediction = lasagne.layers.get_output(l_u1, deterministic=True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
	test_loss = test_loss.mean()

	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)














