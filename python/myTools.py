import myClasses
import theano
import theano.tensor as T
import numpy
import sys
from PIL import Image

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import nolearn
import math
import numpy as np
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum

#from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from skimage import exposure
import skimage
from skimage.transform import rotate
from scipy import ndimage
from matplotlib import gridspec

#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm


import lasagne
from lasagne.layers import InputLayer, Conv2DLayer, DenseLayer, MaxPool2DLayer, InverseLayer, Pool2DLayer
from lasagne.layers import (InputLayer, ConcatLayer, Pool2DLayer, ReshapeLayer, DimshuffleLayer, NonlinearityLayer,
                            DropoutLayer, Deconv2DLayer, batch_norm)
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.init import HeNormal
#from myClasses import Deconv2DLayer


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

	#extracts the id of the imageimport math
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
def cropCenter(images, cropPercentage):
	#get the original dimensions of the images
	originalXdim=images.shape[2]
	originalYdim=images.shape[3]

	#compute the new dimensions based on the cropPercentage
	newXdim=int(math.floor(originalXdim*cropPercentage)/100)
	newYdim=int(math.floor(originalYdim*cropPercentage)/100)

	#compute how many pixels from each side should be cropped
	cropx=int((originalXdim-newXdim)/2)
	cropy=int((originalYdim-newYdim)/2)

	#crop images
	croppedImages=images[:, :, cropx:originalXdim-cropx, cropy:originalYdim-cropy]

	#get actual new image dimensions
	actualNewXdim=croppedImages.shape[2]
	actualNewYdim=croppedImages.shape[3]

	print('--------------------')
	print('Cropped images from %d x %d to %d x %d' % (originalXdim, originalYdim, actualNewXdim, actualNewYdim))

	return croppedImages

def cropCenter1(image, cropPercentage):
	#get the original dimensions of the images
	originalXdim=image.shape[0]
	originalYdim=image.shape[1]

	#compute the new dimensions based on the cropPercentage
	newXdim=originalXdim*cropPercentage/100
	newYdim=originalYdim*cropPercentage/100

	#compute how many pixels from each side should be cropped
	cropx=(originalXdim-newXdim)/2
	cropy=(originalYdim-newYdim)/2

	#crop images
	croppedImage=image[cropx:originalXdim-cropx-1, cropy:originalYdim-cropy-1]

	#get actual new image dimensions
	actualNewXdim=croppedImage.shape[0]
	actualNewYdim=croppedImage.shape[1]

	print('--------------------')
	print('Cropped image from %d x %d to %d x %d' % (originalXdim, originalYdim, actualNewXdim, actualNewYdim))

	return croppedImage

def dt(masks, threshold):

	masks[masks>0]=1

	for x in numpy.nditer(masks, op_flags=['readwrite']):
     		if x==1:
             		x[...]=0
    		else:
	     		x[...]=1

	masks=masks.astype(numpy.float32)

	for i in xrange(masks.shape[0]):
		masks[i][0]=ndimage.distance_transform_edt(masks[i][0])

	masks[masks>threshold]=threshold

	masks=masks/numpy.amax(masks).astype(numpy.float32)
	masks=masks.astype(numpy.float32)

	return masks

def augmentImageStatic(img, numOfTiles=4, overlap=False):

	#rotation angles
	angles=[0]

	#get the size of the image
	imgXsize=img.shape[0]
	imgYsize=img.shape[1]


	if overlap==False:

		#compute the size of the tiles
		tileWidth=int(math.floor((2*imgXsize/numOfTiles)))
		tileHeight=int(math.floor((2*imgXsize/numOfTiles)))

		#preallocate space for the tiles (3 refers to the two different types of mirroring + the normal version)
		tiles=numpy.empty([numOfTiles*len(angles)*3, tileWidth, tileHeight])#440


		bufferIndex=0
		for i in angles:
			#rotate the image
			tempImg=skimage.transform.rotate(img, i)

			tile1=tempImg[0:tileWidth, 0:tileHeight]#tile1=tempImg[0:440, 0:440]
			tile2=tempImg[tileWidth:imgXsize, 0:tileHeight]#tile2=tempImg[410:850, 0:440]
			tile3=tempImg[0:tileWidth, tileHeight:imgYsize]#tile3=tempImg[0:440, 410:850]
			tile4=tempImg[tileWidth:imgXsize, tileHeight:imgYsize]#tile4=tempImg[410:850, 410:850]
			#plt.show(plt.imshow(tile, cmap=cm.binary))
			tiles[bufferIndex]=tile1
			bufferIndex+=1
			tiles[bufferIndex]=tile2
			bufferIndex+=1
			tiles[bufferIndex]=tile3
			bufferIndex+=1
			tiles[bufferIndex]=tile4
			bufferIndex+=1


		#apply mirroring (left-right)
		flipedImg=numpy.fliplr(img)

		for i in angles:
			#rotate the image
			tempImg=skimage.transform.rotate(flipedImg, i)
			tile1=tempImg[0:tileWidth, 0:tileHeight]#tile1=tempImg[0:440, 0:440]
			tile2=tempImg[tileWidth:imgXsize, 0:tileHeight]#tile2=tempImg[410:850, 0:440]
			tile3=tempImg[0:tileWidth, tileHeight:imgYsize]#tile3=tempImg[0:440, 410:850]
			tile4=tempImg[tileWidth:imgXsize, tileHeight:imgYsize]#ile4=tempImg[410:850, 410:850]
			#plt.show(plt.imshow(tile, cmap=cm.binary))
			tiles[bufferIndex]=tile1
			bufferIndex+=1
			tiles[bufferIndex]=tile2
			bufferIndex+=1
			tiles[bufferIndex]=tile3
			bufferIndex+=1
			tiles[bufferIndex]=tile4
			bufferIndex+=1

		#apply mirroring (up-down)
		flipedImg=numpy.flipud(img)

		for i in angles:
			#rotate the image
			tempImg=skimage.transform.rotate(flipedImg, i)
			tile1=tempImg[0:tileWidth, 0:tileHeight]#tile1=tempImg[0:440, 0:440]
			tile2=tempImg[tileWidth:imgXsize, 0:tileHeight]#tile2=tempImg[410:850, 0:440]
			tile3=tempImg[0:tileWidth, tileHeight:imgYsize]#tile3=tempImg[0:440, 410:850]
			tile4=tempImg[tileWidth:imgXsize, tileHeight:imgYsize]#tile4=tempImg[410:850, 410:850]
			#plt.show(plt.imshow(tile, cmap=cm.binary))
			tiles[bufferIndex]=tile1
			bufferIndex+=1
			tiles[bufferIndex]=tile2
			bufferIndex+=1
			tiles[bufferIndex]=tile3
			bufferIndex+=1
			tiles[bufferIndex]=tile4
			bufferIndex+=1


	return tiles

def augmentImage(img, numOfTiles=4, overlap=False):

	#rotation angles
	angles=[0]

	#get the size of the image
	imgXsize=img.shape[0]
	imgYsize=img.shape[1]


	if overlap==False:

		#compute the size of the tiles
		tileWidth=math.floor((imgXsize/numOfTiles)*2)
		tileHeight=math.floor((imgXsize/numOfTiles)*2)

		#preallocate space for the tiles (3 refers to the two different types of mirroring + the normal version)
		tiles=numpy.empty([numOfTiles*len(angles)*3, tileHeight, tileWidth])


		bufferIndex=0
		for i in angles:
			#rotate the image
			tempImg=skimage.transform.rotate(img, i)
			for x in range(numOfTiles/2):
				for y in range(numOfTiles/2):
					tile=tempImg[x*tileWidth:(x+1)*tileWidth, y*tileHeight:(y+1)*tileHeight]
					#plt.show(plt.imshow(tile, cmap=cm.binary))
					tiles[bufferIndex]=tile
					bufferIndex+=1


		#apply mirroring (left-right)
		flipedImg=numpy.fliplr(img)

		for i in angles:
			#rotate the image
			tempImg=skimage.transform.rotate(flipedImg, i)
			for x in range(numOfTiles/2):
				for y in range(numOfTiles/2):
					tile=tempImg[x*tileWidth:(x+1)*tileWidth, y*tileHeight:(y+1)*tileHeight]
					#plt.show(plt.imshow(tile, cmap=cm.binary))
					tiles[bufferIndex]=tile
					bufferIndex+=1

		#apply mirroring (up-down)
		flipedImg=numpy.flipud(img)

		for i in angles:
			#rotate the image
			tempImg=skimage.transform.rotate(flipedImg, i)
			for x in range(numOfTiles/2):
				for y in range(numOfTiles/2):
					tile=tempImg[x*tileWidth:(x+1)*tileWidth, y*tileHeight:(y+1)*tileHeight]
					#plt.show(plt.imshow(tile, cmap=cm.binary))
					tiles[bufferIndex]=tile
					bufferIndex+=1


	if overlap==True:

		#compute the size of the tiles
		tileWidth=math.floor((imgXsize/numOfTiles)*3)
		tileHeight=math.floor((imgXsize/numOfTiles)*3)

		#preallocate space for the tiles (2 refers to the two different types of mirroring)
		tiles=numpy.empty([numOfTiles*len(angles)*2, tileHeight, tileWidth])

		#apply mirroring (left-right)
		flipedImg=numpy.fliplr(img)

		bufferIndex=0
		for i in angles:
			#rotate the image
			tempImg=skimage.transform.rotate(flipedImg, i)
			j=0
			k=tileWidth
			l=0
			m=tileHeight
			for x in range(numOfTiles/2):
				for y in range(numOfTiles/2):

					tile=tempImg[j:k,l:m]
					#plt.show(plt.imshow(tile, cmap=cm.binary))
					tiles[bufferIndex]=tile
					bufferIndex+=1
					l+=tileWidth/numOfTiles
					m+=tileWidth/numOfTiles
				l=0
				m=tileHeight
				j+=tileWidth/numOfTiles
				k+=tileWidth/numOfTiles

		#apply mirroring (up-down)
		flipedImg=numpy.flipud(img)

		for i in angles:
			#rotate the image
			tempImg=skimage.transform.rotate(flipedImg, i)
			j=0
			k=tileWidth
			l=0
			m=tileHeight
			for x in range(numOfTiles/2):
				for y in range(numOfTiles/2):

					tile=tempImg[j:k,l:m]
					#plt.show(plt.imshow(tile, cmap=cm.binary))
					tiles[bufferIndex]=tile
					bufferIndex+=1
					l+=tileWidth/numOfTiles
					m+=tileWidth/numOfTiles
				l=0
				m=tileHeight
				j+=tileWidth/numOfTiles
				k+=tileWidth/numOfTiles

	return tiles



def augmentImage1(img, numOfTiles=1):

	#rotation angles
	angles=[0, 90]

	#get the size of the image
	imgXsize=img.shape[0]
	imgYsize=img.shape[1]

	#compute the size of the tiles
	tileWidth=imgXsize
	tileHeight=imgYsize

	#preallocate space for the tiles (2 refers to the two different types of mirroring)
	tiles=numpy.empty([numOfTiles*len(angles)*3, tileHeight, tileWidth])

	bufferIndex=0
	for i in angles:
		#rotate the image
		tempImg=skimage.transform.rotate(img, i)
		tiles[bufferIndex]=tempImg
		bufferIndex+=1


	#apply mirroring (left-right)
	flipedImg=numpy.fliplr(img)

	for i in angles:
		#rotate the image
		tempImg=skimage.transform.rotate(flipedImg, i)
		tiles[bufferIndex]=tempImg
		bufferIndex+=1

	#apply mirroring (up-down)
	flipedImg=numpy.flipud(img)

	for i in angles:
		#rotate the image
		tempImg=skimage.transform.rotate(flipedImg, i)
		tiles[bufferIndex]=tempImg
		bufferIndex+=1

	return tiles



def augmentData(dataset, numOfTiles, overlap, imageWidth, imageHeight):

	print('--------------------')
	print('Augmenting the dataset...')
	start_time = time.time()


	if overlap==True:
		#compute the size of the tiles
		tileWidth=math.floor((imageWidth/numOfTiles)*3)
		tileHeight=math.floor((imageHeight/numOfTiles)*3)
	else:
		#compute the size of the tiles
		tileWidth=int(math.floor(2*imageWidth/numOfTiles))
		tileHeight=int(math.floor(2*imageHeight/numOfTiles))

	#if we do not want tiling
	if numOfTiles==1:
		tileWidth=imageWidth
		tileHeight=imageHeight

	#preallocate space for the dataset (4 refers to the number of the rotation angles, 3 refers to the types of mirroring + the normal version)
	augmented=numpy.empty([dataset.shape[0]*numOfTiles*2*3, tileWidth, tileHeight])

	bufferIndex=0
	for i in range(dataset.shape[0]):
		#if we want tiling, then use the augmentImage function
		if numOfTiles!=1:
			newImgs=augmentImageStatic(dataset[i][0], numOfTiles, overlap)
		else:
			newImgs=augmentImage1(dataset[i][0])

		for j in range(newImgs.shape[0]):
			augmented[bufferIndex]=newImgs[j]
			bufferIndex+=1

	end_time = time.time()
	print('Augmented %d images in %.2f seconds' % (dataset.shape[0], end_time-start_time))

	return augmented.reshape(augmented.shape[0], 1, augmented.shape[1], augmented.shape[2])


def augmentMasks(dataset, numOfTiles, overlap, imageWidth, imageHeight):

	print('--------------------')
	print('Augmenting the dataset...')
	start_time = time.time()


	if overlap==True:
		#compute the size of the tiles
		tileWidth=math.floor((imageWidth/numOfTiles)*3)
		tileHeight=math.floor((imageHeight/numOfTiles)*3)
	else:
		#compute the size of the tiles
		tileWidth=math.floor((imageWidth/numOfTiles)*2)
		tileHeight=math.floor((imageHeight/numOfTiles)*2)

	#if we do not want tiling
	if numOfTiles==1:
		tileWidth=imageWidth
		tileHeight=imageHeight

	#preallocate space for the dataset (4 refers to the number of the rotation angles, 3 refers to the types of mirroring + the normal version)
	augmented=numpy.empty([dataset.shape[0]*numOfTiles*1*3, tileWidth, tileHeight])

	bufferIndex=0
	for i in range(dataset.shape[0]):
		#if we want tiling, then use the augmentImage function
		if numOfTiles!=1:
			newImgs=augmentImage(dataset[i][0], numOfTiles, overlap)
		else:
			newImgs=augmentImage1(dataset[i][0])

		for j in range(newImgs.shape[0]):
			augmented[bufferIndex]=newImgs[j]
			bufferIndex+=1

	end_time = time.time()
	print('Augmented %d images in %.2f seconds' % (dataset.shape[0], end_time-start_time))

	return augmented.reshape(augmented.shape[0], 1, augmented.shape[1], augmented.shape[2])


#applies histogram equalisation to images
def myHistEq(images):

	print('--------------------')
	print('Performing Histogram Equalisation...')
	start_time = time.time()

	#image values are expected to be float numbers
	images=images.astype(theano.config.floatX)

	#apply histogram equalisation
	images=exposure.equalize_hist(images)

	end_time = time.time()
	print('Histogram Equalisation completed in %.2f seconds' % (end_time-start_time))

	return images


#applies mean normalisation to images
def myMeanNorm(images):

	print('--------------------')
	print('Performing Mean Normalisation...')
	start_time = time.time()

	#image values are expected to be float numbers
	images=images.astype(theano.config.floatX)

	minValue=images.min()
	maxValue=images.max()

	images -= minValue
	images *=(255.0/(maxValue-minValue))

	end_time = time.time()
	print('Mean Normalisation completed in %.2f seconds' % (end_time-start_time))

	return images


#applies contrast streching to images
def myContrStrech(images):

	print('--------------------')
	print('Performing Contrast Streching...')
	start_time = time.time()

	#image values are expected to be float numbers
	images=images.astype(theano.config.floatX)

	p5, p95 = np.percentile(images, (5, 95))
	img_rescale = exposure.rescale_intensity(images, in_range=(p5, p95))

	end_time = time.time()
	print('Contrast Streching completed in %.2f seconds' % (end_time-start_time))

	return img_rescale


#creates a convolutional neural network
def createNN(data_size, X, Y, valX, valY, epochs, n_batches, batch_size, learning_rate, w_decay):

	#creating symbolic variables for input and output
	input_var = T.tensor4('input')
	target_var = T.tensor4('targets')
	#initialising an empty network
	net = {}
	base_n_filters=4

	#Input layer:
	net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)

	#the rest of the network structure
	#net['conv00000'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['data'], num_filters=30, filter_size=7))
	#net['conv0000'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['conv00000'], num_filters=30, filter_size=6))
	#net['conv000'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['conv0000'], num_filters=30, filter_size=6))
	#net['conv00'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['data'], num_filters=35, filter_size=6))
	#net['conv0'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['conv00'], num_filters=35, filter_size=6))
	net['conv1'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['data'], num_filters=32, filter_size=5))
	net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'], pool_size=2)
	net['conv2'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool1'], num_filters=32, filter_size=5))
	net['pool2'] = lasagne.layers.Pool2DLayer(net['conv2'], pool_size=2)
	net['conv3'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool2'], num_filters=32, filter_size=5))
	net['pool3'] = lasagne.layers.Pool2DLayer(net['conv3'], pool_size=2)
	net['conv4'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool3'], num_filters=32, filter_size=5))
	net['deconv4']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['conv4'], num_filters=32, filter_size=5))
	net['unpool3']= lasagne.layers.InverseLayer(net['deconv4'], net['pool3'])
	net['deconv3']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool3'], num_filters=32, filter_size=5))
	net['unpool2']= lasagne.layers.InverseLayer(net['deconv3'], net['pool2'])
	net['deconv2']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool2'], num_filters=32, filter_size=5))
	net['unpool1']= lasagne.layers.InverseLayer(net['deconv2'], net['pool1'])
	net['deconv1']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool1'], num_filters=32, filter_size=5))
	net['output']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['deconv1'], num_filters=1, filter_size=1, nonlinearity=lasagne.nonlinearities.sigmoid))



	print('--------------------')
	print('Network architecture: \n')

	#get all layers
	allLayers=lasagne.layers.get_all_layers(net['output'])
	#for each layer print its shape information
	for l in allLayers:
		print(lasagne.layers.get_output_shape(l))


	#print the total number of trainable parameters of the network
	print('\nThe total number of trainable parameters is %d' % (lasagne.layers.count_params(net['output'])))
	print('\nTraining on %d images' % (X.shape[0]))

	#with np.load('model.npz') as f:
	#	param_values = [f['arr_%d' % i] for i in range(len(f.files))]

	#lasagne.layers.set_all_param_values(net['output'], param_values)

	myNet=net['output']

	lr = learning_rate
	weight_decay = w_decay

	#define how to get the prediction of the network
	prediction = lasagne.layers.get_output(myNet)

	#define the cost function
	#loss = lasagne.objectives.squared_error(prediction, target_var)
	#loss = loss.mean()
	loss = myCrossEntropy(prediction, target_var)
	loss = loss.mean()
	#also add weight decay to the cost function
	weightsl2 = lasagne.regularization.regularize_network_params(myNet, lasagne.regularization.l2)
	loss += weight_decay * weightsl2

	#get all the trainable parameters of the network
	params = lasagne.layers.get_all_params(myNet, trainable=True)

	#define the update function for each training step
	updates = lasagne.updates.adam(loss, params, learning_rate=lr)

	#compile a train function
	train_fn = theano.function([input_var, target_var], loss, updates=updates)

 	#defining same things for testing
	##"deterministic=True" disables stochastic behaviour, such as dropout
	test_prediction = lasagne.layers.get_output(myNet, deterministic=True)
	test_loss = myCrossEntropy(test_prediction, target_var)
	test_loss = test_loss.mean()
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

	#compile a theano validation function
	val_fn = theano.function([input_var, target_var], test_loss)

	#compile a theano function to make predictions with the network
	get_preds = theano.function([input_var], test_prediction)

	####### the actual training ########
	print('--------------------')
	#get the number of training examples
	n_examples = X.shape[0]

	start_time = time.time()

	cost_history=[]

	#for each epoch train for all the batches
	for epoch in xrange(epochs):
		epoch_time_start=time.time()
		batch_cost_history=[]

		#for each batch train and update the weights
		for batch in xrange(n_batches):
			x_batch = X[batch*batch_size: (batch+1) * batch_size]
			y_batch = Y[batch*batch_size: (batch+1) * batch_size]

			this_cost = train_fn(x_batch, y_batch)

			batch_cost_history.append(this_cost)

		epoch_cost = np.mean(batch_cost_history)
		cost_history.append(epoch_cost)

		#spliting the calculation of the test loss to half, so that it does not waste much memory
		test_cost=0
		for i in xrange(valX.shape[0]):
			test_cost+=val_fn(np.reshape(valX[i,:,:,:], (1,1,valX.shape[2],valX.shape[3])),np.reshape(valY[i,:,:,:],(1,1,valY.shape[2],valY.shape[3])))
		test_cost = np.float32(test_cost/valX.shape[0])
		epoch_time_end = time.time()
		print('Epoch %d/%d, train error: %f, val error: %f. Elapsed time: %.2f s' % (epoch+1, epochs, epoch_cost, test_cost, epoch_time_end-epoch_time_start))

	end_time = time.time()
	print('Training completed in %.2f seconds.' % (end_time - start_time))


	#for each layer print the resulted filters
	#for l in range(1, len(allLayers)):
	#	if isinstance(allLayers[l], Conv2DLayer):
	#		visualize.plot_conv_weights(allLayers[l])


	numpy.savez('model.npz', *lasagne.layers.get_all_param_values(net['output']))
	return get_preds

#creates a convolutional neural network
def createUnet(data_size, X, Y, valX, valY, epochs, n_batches, batch_size, learning_rate, w_decay):

	#creating symbolic variables for input and output
	input_var = T.tensor4('input')
	target_var = T.tensor4('targets')
	#initialising an empty network
	net = {}
	base_n_filters=64
	do_dropout=True
	nonlinearity=lasagne.nonlinearities.rectify
	pad='same'

	#Input layer:
	net['input'] = InputLayer(data_size, input_var=input_var)

	net['contr_1_1'] = batch_norm(ConvLayer(net['input'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad))
	net['contr_1_2'] = batch_norm(ConvLayer(net['contr_1_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad))
	net['pool1'] = Pool2DLayer(net['contr_1_2'], 2)

	net['contr_2_1'] = batch_norm(ConvLayer(net['pool1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad))
	net['contr_2_2'] = batch_norm(ConvLayer(net['contr_2_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad))
	net['pool2'] = Pool2DLayer(net['contr_2_2'], 2)

	net['contr_3_1'] = batch_norm(ConvLayer(net['pool2'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad))
	net['contr_3_2'] = batch_norm(ConvLayer(net['contr_3_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad))
	net['pool3'] = Pool2DLayer(net['contr_3_2'], 2)

	net['contr_4_1'] = batch_norm(ConvLayer(net['pool3'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad))
	net['contr_4_2'] = batch_norm(ConvLayer(net['contr_4_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad))
	l = net['pool4'] = Pool2DLayer(net['contr_4_2'], 2)
	# the paper does not really describe where and how dropout is added. Feel free to try more options
	if do_dropout:
		l = DropoutLayer(l, p=0.1)

	net['encode_1'] = batch_norm(ConvLayer(l, base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad))
	net['encode_2'] = batch_norm(ConvLayer(net['encode_1'], base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad))
	net['upscale1'] = batch_norm(Deconv2DLayer(net['encode_2'], base_n_filters*16, 2, 2, crop="valid", nonlinearity=nonlinearity))

	net['concat1'] = ConcatLayer([net['upscale1'], net['contr_4_2']], cropping=(None, None, "center", "center"))
	net['expand_1_1'] = batch_norm(ConvLayer(net['concat1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad))
	net['expand_1_2'] = batch_norm(ConvLayer(net['expand_1_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad))
	net['upscale2'] = batch_norm(Deconv2DLayer(net['expand_1_2'], base_n_filters*8, 2, 2, crop="valid", nonlinearity=nonlinearity))

	net['concat2'] = ConcatLayer([net['upscale2'], net['contr_3_2']], cropping=(None, None, "center", "center"))
	net['expand_2_1'] = batch_norm(ConvLayer(net['concat2'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad))
	net['expand_2_2'] = batch_norm(ConvLayer(net['expand_2_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad))
	net['upscale3'] = batch_norm(Deconv2DLayer(net['expand_2_2'], base_n_filters*4, 2, 2, crop="valid", nonlinearity=nonlinearity))

	net['concat3'] = ConcatLayer([net['upscale3'], net['contr_2_2']], cropping=(None, None, "center", "center"))
	net['expand_3_1'] = batch_norm(ConvLayer(net['concat3'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad))
	net['expand_3_2'] = batch_norm(Deconv2DLayer(net['expand_3_1'], base_n_filters*2, 3, nonlinearity=nonlinearity))#, pad=pad))
	net['upscale4'] = batch_norm(Deconv2DLayer(net['expand_3_2'], base_n_filters*2, 2, 2, crop="valid", nonlinearity=nonlinearity))

	net['concat4'] = ConcatLayer([net['upscale4'], net['contr_1_2']], cropping=(None, None, "center", "center"))
	net['expand_4_1'] = batch_norm(Deconv2DLayer(net['concat4'], base_n_filters, 3, nonlinearity=nonlinearity))#, pad=pad))
	net['expand_4_2'] = batch_norm(Deconv2DLayer(net['expand_4_1'], base_n_filters, 3, nonlinearity=nonlinearity))#, pad=pad))

	net['output'] = batch_norm(Deconv2DLayer(net['expand_4_2'], 1, 2, nonlinearity=lasagne.nonlinearities.sigmoid))



	print('--------------------')
	print('Network architecture: \n')

	#get all layers
	allLayers=lasagne.layers.get_all_layers(net['output'])
	#for each layer print its shape information
	for l in allLayers:
		print(lasagne.layers.get_output_shape(l))


	#print the total number of trainable parameters of the network
	print('\nThe total number of trainable parameters is %d' % (lasagne.layers.count_params(net['output'])))
	print('\nTraining on %d images' % (X.shape[0]))

	#with np.load('model.npz') as f:
	#	param_values = [f['arr_%d' % i] for i in range(len(f.files))]

	#lasagne.layers.set_all_param_values(net['output'], param_values)

	myNet=net['output']

	lr = learning_rate
	weight_decay = w_decay

	#define how to get the prediction of the network
	prediction = lasagne.layers.get_output(myNet)

	#define the cost function
	#loss = lasagne.objectives.squared_error(prediction, target_var)
	#loss = loss.mean()
	loss = myCrossEntropy(prediction, target_var)
	loss = loss.mean()
	#also add weight decay to the cost function
	weightsl2 = lasagne.regularization.regularize_network_params(myNet, lasagne.regularization.l2)
	loss += weight_decay * weightsl2

	#get all the trainable parameters of the network
	params = lasagne.layers.get_all_params(myNet, trainable=True)

	#define the update function for each training step
	updates = lasagne.updates.adam(loss, params, learning_rate=lr)

	#compile a train function
	train_fn = theano.function([input_var, target_var], loss, updates=updates)

 	#defining same things for testing
	##"deterministic=True" disables stochastic behaviour, such as dropout
	test_prediction = lasagne.layers.get_output(myNet, deterministic=True)
	test_loss = myCrossEntropy(test_prediction, target_var)
	test_loss = test_loss.mean()
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

	#compile a theano validation function
	val_fn = theano.function([input_var, target_var], test_loss)

	#compile a theano function to make predictions with the network
	get_preds = theano.function([input_var], test_prediction)

	####### the actual training ########
	print('--------------------')
	#get the number of training examples
	n_examples = X.shape[0]

	start_time = time.time()

	cost_history=[]

	#for each epoch train for all the batches
	for epoch in xrange(epochs):
		epoch_time_start=time.time()
		batch_cost_history=[]

		#for each batch train and update the weights
		for batch in xrange(n_batches):
			x_batch = X[batch*batch_size: (batch+1) * batch_size]
			y_batch = Y[batch*batch_size: (batch+1) * batch_size]

			this_cost = train_fn(x_batch, y_batch)

			batch_cost_history.append(this_cost)

		epoch_cost = np.mean(batch_cost_history)
		cost_history.append(epoch_cost)

		#spliting the calculation of the test loss to half, so that it does not waste much memory
		test_cost=0
		for i in xrange(valX.shape[0]):
			test_cost+=val_fn(np.reshape(valX[i,:,:,:], (1,1,valX.shape[2],valX.shape[3])),np.reshape(valY[i,:,:,:],(1,1,valY.shape[2],valY.shape[3])))
		test_cost = np.float32(test_cost/valX.shape[0])
		epoch_time_end = time.time()
		print('Epoch %d/%d, train error: %f, val error: %f. Elapsed time: %.2f s' % (epoch+1, epochs, epoch_cost, test_cost, epoch_time_end-epoch_time_start))

	end_time = time.time()
	print('Training completed in %.2f seconds.' % (end_time - start_time))


	#for each layer print the resulted filters
	#for l in range(1, len(allLayers)):
	#	if isinstance(allLayers[l], Conv2DLayer):
	#		visualize.plot_conv_weights(allLayers[l])


	numpy.savez('model.npz', *lasagne.layers.get_all_param_values(net['output']))
	return get_preds


def createPretrainedNN(data_size):

	#creating symbolic variables for input and output
	input_var = T.tensor4('input')
	target_var = T.tensor4('targets')
	#initialising an empty network
	net = {}

	#Input layer:
	net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)

	#the rest of the network structure
	#net['conv00000'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['data'], num_filters=30, filter_size=7))
	#net['conv0000'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['conv00000'], num_filters=30, filter_size=6))
	#net['conv000'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['conv0000'], num_filters=30, filter_size=6))
	#net['conv00'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['data'], num_filters=35, filter_size=6))
	#net['conv0'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['conv00'], num_filters=35, filter_size=6))
	net['conv1'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['data'], num_filters=35, filter_size=5))
	net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'], pool_size=2)
	net['conv2'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool1'], num_filters=35, filter_size=5))
	net['pool2'] = lasagne.layers.Pool2DLayer(net['conv2'], pool_size=2)
	net['conv3'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool2'], num_filters=35, filter_size=5))
	net['pool3'] = lasagne.layers.Pool2DLayer(net['conv3'], pool_size=2)
	net['conv4'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool3'], num_filters=35, filter_size=5))
	net['deconv4']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['conv4'], num_filters=35, filter_size=5))
	net['unpool3']= lasagne.layers.InverseLayer(net['deconv4'], net['pool3'])
	net['deconv3']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool3'], num_filters=35, filter_size=5))
	net['unpool2']= lasagne.layers.InverseLayer(net['deconv3'], net['pool2'])
	net['deconv2']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool2'], num_filters=35, filter_size=5))
	net['unpool1']= lasagne.layers.InverseLayer(net['deconv2'], net['pool1'])
	net['deconv1']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool1'], num_filters=35, filter_size=5))
	net['output']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['deconv1'], num_filters=1, filter_size=1, nonlinearity=lasagne.nonlinearities.sigmoid))

	with np.load('modelbest.npz') as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]

	lasagne.layers.set_all_param_values(net['output'], param_values)

	myNet=net['output']

	test_prediction = lasagne.layers.get_output(myNet, deterministic=True)

	#compile a theano function to make predictions with the network
	get_preds = theano.function([input_var], test_prediction)

	allLayers=lasagne.layers.get_all_layers(net['output'])
	#layer = allLayers[1]
	#W = layer.W.get_value()
	#b = layer.b.get_value()
	#f = [w + bb for w, bb in zip(W, b)]

	#gs = gridspec.GridSpec(6, 6)
	#for i in range(layer.num_filters):
	#	g = gs[i]
	#	ax = plt.subplot(g)
	#	ax.grid()
	#	ax.set_xticks([])
	#	ax.set_yticks([])
	#	ax.imshow(W[i][0],interpolation='nearest',cmap=cm.binary)
	#plt.show(gs)
	#for each layer print the resulted filters
	#for l in xrange(len(allLayers)):
	#	if isinstance(allLayers[l], Conv2DLayer):
	#plt.show(visualize.plot_conv_weights(allLayers[20]))


	return get_preds

#same ase createPretrainedNN, but with arguments
def createPretrainedNN2(data_size, filters, modelFile):

	filters=int(filters)

	#creating symbolic variables for input and output
	input_var = T.tensor4('input')
	target_var = T.tensor4('targets')
	#initialising an empty network
	net = {}

	#Input layer:
	net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)

	#the rest of the network structure
	#net['conv00000'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['data'], num_filters=30, filter_size=7))
	#net['conv0000'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['conv00000'], num_filters=30, filter_size=6))
	#net['conv000'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['conv0000'], num_filters=30, filter_size=6))
	#net['conv00'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['data'], num_filters=35, filter_size=6))
	#net['conv0'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['conv00'], num_filters=35, filter_size=6))
	net['conv1'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['data'], num_filters=filters, filter_size=5))
	net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'], pool_size=2)
	net['conv2'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool1'], num_filters=filters, filter_size=5))
	net['pool2'] = lasagne.layers.Pool2DLayer(net['conv2'], pool_size=2)
	net['conv3'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool2'], num_filters=filters, filter_size=5))
	net['pool3'] = lasagne.layers.Pool2DLayer(net['conv3'], pool_size=2)
	net['conv4'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool3'], num_filters=filters, filter_size=5))
	net['deconv4']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['conv4'], num_filters=filters, filter_size=5))
	net['unpool3']= lasagne.layers.InverseLayer(net['deconv4'], net['pool3'])
	net['deconv3']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool3'], num_filters=filters, filter_size=5))
	net['unpool2']= lasagne.layers.InverseLayer(net['deconv3'], net['pool2'])
	net['deconv2']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool2'], num_filters=filters, filter_size=5))
	net['unpool1']= lasagne.layers.InverseLayer(net['deconv2'], net['pool1'])
	net['deconv1']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool1'], num_filters=filters, filter_size=5))
	net['output']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['deconv1'], num_filters=1, filter_size=1, nonlinearity=lasagne.nonlinearities.sigmoid))

	with np.load(modelFile) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]

	lasagne.layers.set_all_param_values(net['output'], param_values)

	myNet=net['output']

	test_prediction = lasagne.layers.get_output(myNet, deterministic=True)

	#compile a theano function to make predictions with the network
	get_preds = theano.function([input_var], test_prediction)

	allLayers=lasagne.layers.get_all_layers(net['output'])
	#layer = allLayers[1]
	#W = layer.W.get_value()
	#b = layer.b.get_value()
	#f = [w + bb for w, bb in zip(W, b)]

	#gs = gridspec.GridSpec(6, 6)
	#for i in range(layer.num_filters):
	#	g = gs[i]
	#	ax = plt.subplot(g)
	#	ax.grid()
	#	ax.set_xticks([])
	#	ax.set_yticks([])
	#	ax.imshow(W[i][0],interpolation='nearest',cmap=cm.binary)
	#plt.show(gs)
	#for each layer print the resulted filters
	#for l in xrange(len(allLayers)):
	#	if isinstance(allLayers[l], Conv2DLayer):
	#plt.show(visualize.plot_conv_weights(allLayers[20]))


	return get_preds

def trainNN(myNet, X, Y, epochs, n_batches, batch_size):

	#creating symbolic variables for input and output
	input_var = T.tensor4('input')
	target_var = T.tensor4('targets')

	lr = 0.01
	moment=0.9
	weight_decay = 0.0005

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
	updates = lasagne.updates.momentum(loss, params, learning_rate=lr, momentum=moment)

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



def myCostFunction(a, b):


    r=np.float32(0.04)

    sensitivity=(1-r)*T.sum(((b - a)**2)*b)/T.sum(b)

    specificity=r*T.sum(((b - a)**2)*(1-b))/T.sum(1-b)

    return sensitivity+specificity


def myTestCostFunction(a, b):


    r=np.float32(0.04)

    sensitivity=(1-r)*numpy.sum(((b - a)**2)*b)/numpy.sum(b)

    specificity=r*numpy.sum(((b - a)**2)*(1-b))/numpy.sum(1-b)

    return sensitivity+specificity


def myCrossEntropy(predictions, targets):

	r=np.float32(0.80)

	#myFactor=T.sum(predictions>0.3 and predictions<0.7)/T.sum(predictions<2.0)

	return -1*targets*T.log(predictions)*1.0+(-1)*(1-targets)*T.log(1-predictions)*0.5


def myTestCrossEntropy(predictions, targets):

	r=np.float32(0.06)

	return numpy.average(-1*targets*numpy.log(predictions)*(1-r)+(-1)*(1-targets)*numpy.log(1-predictions)*r)
