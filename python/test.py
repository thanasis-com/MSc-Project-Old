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



dataSet=myTools.loadImages('/home/athanasiostsiaras/Downloads/images', 1024, 1024, 4)

dataSet=myTools.oneDimension(dataSet)
	
#plt.show(plt.imshow(dataSet[0][0], cmap=cm.binary))

masks=myTools.loadImages('/home/athanasiostsiaras/Downloads/masks', 819, 819, 1)

#plt.show(plt.imshow(masks[0][0], cmap=cm.binary))

temp=myTools.crop(dataSet, 80)
test=temp[21:40, :, :, :]
train=temp[0:20, :, :, :]


for x in numpy.nditer(masks, op_flags=['readwrite']):
     if x>0:
             x[...]=1

#plt.show(plt.imshow(masks[0][0], cmap=cm.binary))

masks=masks.astype(numpy.int32)

#temp=numpy.random.rand(12,1,819,819)
#masks=numpy.random.random_integers(0,1, size=(12,1,819,819)).astype(numpy.int32) 

data_size=(None,1,819,819)

myNet=myTools.createNN(data_size, X=train, Y=masks[0:20, :, :, :], epochs=2, n_batches=20, batch_size=1)

res=myNet(test)

plt.show(plt.imshow(test[0][0], cmap=cm.binary))

plt.show(plt.imshow(res[0][0], cmap=cm.binary))

sys.exit()



#loss, acc = val_fn(temp, masks)
#test_error = 1 - acc
#print('Test error: %f' % test_error)

#rand=numpy.random.rand(10,1,819,819)
#randMasks=numpy.random.random_integers(0,1, size=(10,1,819,819))
#randMasks=numpy.random.random_sample(size=(41,1,819,819))






