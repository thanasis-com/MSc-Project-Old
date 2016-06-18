import myClasses
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
	
	print('--------------------')
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


#crops the center piece of an image. cropPercentage defines the size of the resulting image
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

	print('--------------------')
	print('Cropped images from %d x %d to %d x %d' % (originalXdim, originalYdim, actualNewXdim, actualNewYdim))	

	return croppedImages
 

#creates a convolutional neural network 
def createNN(data_size, X, Y, epochs, n_batches, batch_size):

	#creating symbolic variables for input and output
	input_var = T.tensor4('input')
	target_var = T.tensor4('targets')	
	#initialising an empty network
	net = {}
	
	#Input layer:
	net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)

	#the rest of the network structure
	net['conv1'] = lasagne.layers.Conv2DLayer(net['data'], num_filters=10, filter_size=5)
	net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'], pool_size=2)
	net['conv2'] = lasagne.layers.Conv2DLayer(net['pool1'], num_filters=10, filter_size=5)
	net['pool2'] = lasagne.layers.Pool2DLayer(net['conv2'], pool_size=2)
	net['unpool2']=lasagne.layers.InverseLayer(net['pool2'], net['pool2'])
	net['deconv2']=myClasses.Deconv2DLayer(net['unpool2'], num_filters=10, filter_size=5)
	net['unpool1']=lasagne.layers.InverseLayer(net['deconv2'], net['pool1'])
	net['output']=myClasses.Deconv2DLayer(net['unpool1'], num_filters=1, filter_size=5, nonlinearity=lasagne.nonlinearities.sigmoid)

	print('--------------------')
	print('Network architecture: \n')
	
	#get all layers
	allLayers=lasagne.layers.get_all_layers(net['output'])
	#for each layer print its shape information
	for l in allLayers:
		print(lasagne.layers.get_output_shape(l))
		

	#print the total number of trainable parameters of the network
	print('\nThe total number of trainable parameters is %d' % (lasagne.layers.count_params(net['output'])))

	myNet=net['output']
	
	lr = 0.1
	weight_decay = 1e-5
	
	#define how to get the prediction of the network
	prediction = lasagne.layers.get_output(myNet)

	#define the cost function
	loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
	loss = loss.mean()
	#also add weight decay to the cost function
	weightsl2 = lasagne.regularization.regularize_network_params(myNet, lasagne.regularization.l2)
	loss += weight_decay * weightsl2

	#get all the trainable parameters of the network
	params = lasagne.layers.get_all_params(myNet, trainable=True)

	#define the update function for each training step
	updates = lasagne.updates.sgd(loss, params, learning_rate=lr)

	#compile a train function
	train_fn = theano.function([input_var, target_var], loss, updates=updates)

 	#defining same things for testing
	##"deterministic=True" disables stochastic behaviour, such as dropout
	test_prediction = lasagne.layers.get_output(myNet, deterministic=True)
	test_loss = lasagne.objectives.binary_crossentropy(test_prediction,target_var)
	test_loss = test_loss.mean()
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

	#compile a theano validation function
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

	#compile a theano function to make predictions with the network
	get_preds = theano.function([input_var], test_prediction)

	####### the actual training ########
	print('--------------------')
	#get the number of training examples
	n_examples = X.shape[0]

	start_time = time.time()

	cost_history=[]
	batch_cost_history=[]

	#for each epoch train for all the batches
	for epoch in xrange(epochs):
		epoch_time_start=time.time()
		#for each batch train and update the weights
    		for batch in xrange(n_batches):
        		x_batch = X[batch*batch_size: (batch+1) * batch_size]
        		y_batch = Y[batch*batch_size: (batch+1) * batch_size]
        
        		this_cost = train_fn(x_batch, y_batch)
	
			batch_cost_history.append(this_cost)

    		epoch_cost = np.mean(batch_cost_history)
    		cost_history.append(epoch_cost)
    		epoch_time_end = time.time()
    		print('Epoch %d/%d, train error: %f. Elapsed time: %.2f seconds' % (epoch+1, epochs, epoch_cost, epoch_time_end-epoch_time_start))

	end_time = time.time()
	print('Training completed in %.2f seconds.' % (end_time - start_time))


	return get_preds	



def trainNN(myNet, X, Y, epochs, n_batches, batch_size):

	#creating symbolic variables for input and output
	input_var = T.tensor4('input')
	target_var = T.tensor4('targets')
	
	lr = 0.1
	weight_decay = 1e-5
	
	#define how to get the prediction of the network
	prediction = lasagne.layers.get_output(myNet)

	#define the cost function
	loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
	loss = loss.mean()
	#also add weight decay to the cost function
	weightsl2 = lasagne.regularization.regularize_network_params(myNet, lasagne.regularization.l2)
	loss += weight_decay * weightsl2

	#get all the trainable parameters of the network
	params = lasagne.layers.get_all_params(myNet, trainable=True)

	#define the update function for each training step
	updates = lasagne.updates.sgd(loss, params, learning_rate=lr)

	#compile a train function
	train_fn = theano.function([input_var, target_var], loss, updates=updates)

 	#defining same things for testing
	##"deterministic=True" disables stochastic behaviour, such as dropout
	test_prediction = lasagne.layers.get_output(myNet, deterministic=True)
	test_loss = lasagne.objectives.binary_crossentropy(test_prediction,target_var)
	test_loss = test_loss.mean()
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

	#compile a theano validation function
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

	#compile a theano function to make predictions with the network
	get_preds = theano.function([input_var], test_prediction)

	####### the actual training ########

	#get the number of training examples
	n_examples = X.shape[0]

	start_time = time.time()

	cost_history=[]
	batch_cost_history=[]

	#for each epoch train for all the batches
	for epoch in xrange(epochs):
		epoch_time_start=time.time()
		#for each batch train and update the weights
    		for batch in xrange(n_batches):
        		x_batch = X[batch*batch_size: (batch+1) * batch_size]
        		y_batch = Y[batch*batch_size: (batch+1) * batch_size]
        
        		this_cost = train_fn(x_batch, y_batch)
	
			batch_cost_history.append(this_cost)

    		epoch_cost = np.mean(batch_cost_history)
    		cost_history.append(epoch_cost)
    		epoch_time_end = time.time()
    		print('Epoch %d/%d, train error: %f. Elapsed time: %.2f seconds' % (epoch+1, epochs, epoch_cost, epoch_time_end-epoch_time_start))

	end_time = time.time()
	print('Training completed in %.2f seconds.' % (end_time - start_time))


	return get_preds








