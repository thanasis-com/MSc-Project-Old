import myTools
import myClasses

import theano
import numpy
from PIL import Image

import time
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import theano.tensor as T
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

from lasagne.layers import InputLayer, Conv2DLayer, DenseLayer, MaxPool2DLayer, InverseLayer



dataSet=myTools.loadImages('../../images', 1024, 1024, 4)

dataSet=myTools.oneDimension(dataSet)

#dataSet=dataSet.astype(numpy.float32)

dataSet= dataSet.astype(numpy.uint8)
	
#plt.show(plt.imshow(dataSet[0][0], cmap=cm.binary))

masks=myTools.loadImages('../../masks', 819, 819, 1)

#plt.show(plt.imshow(masks[0][0], cmap=cm.binary))

dataSet=myTools.cropCenter(dataSet, 80)


test=dataSet[39:40, :, :, :]
train=dataSet[0:38, :, :, :]

dataSet=None

for x in numpy.nditer(masks, op_flags=['readwrite']):
     if x>0:
             x[...]=1

#plt.show(plt.imshow(masks[0][0], cmap=cm.binary))
#plt.show(plt.imshow(train[0][0], cmap=cm.binary))

masks=masks.astype(numpy.float32)

#temp=numpy.random.rand(12,1,819,819)
#masks=numpy.random.random_integers(0,1, size=(12,1,819,819)).astype(numpy.int32) 

data_size=(None,1,819,819)

myNet=myTools.createNN(data_size, X=train, Y=masks[0:38, :, :, :], epochs=1, n_batches=13, batch_size=3)




trainInstance=train[0]
trainInstance=trainInstance.reshape(1, trainInstance.shape[0], trainInstance.shape[1], trainInstance.shape[2])

res=myNet(trainInstance)

#plt.show(plt.imshow(train[0][0], cmap=cm.binary))

#plt.show(plt.imshow(res[0][0], cmap=cm.binary))


testInstance=test[0]
testInstance=testInstance.reshape(1, testInstance.shape[0], testInstance.shape[1], testInstance.shape[2])

res=myNet(testInstance)

#plt.show(plt.imshow(test[0][0], cmap=cm.binary))

#plt.show(plt.imshow(res[0][0], cmap=cm.binary))

#sys.exit()



#loss, acc = val_fn(temp, masks)
#test_error = 1 - acc
#print('Test error: %f' % test_error)

#rand=numpy.random.rand(10,1,819,819)
#randMasks=numpy.random.random_integers(0,1, size=(10,1,819,819))
#randMasks=numpy.random.random_sample(size=(41,1,819,819))







