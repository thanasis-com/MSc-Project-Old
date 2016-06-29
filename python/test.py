import myTools
import myClasses

import theano
import numpy
from PIL import Image

import time
import sys
import math
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


### DATASET
dataSet=myTools.loadImages('../../images', 1024, 1024, 4)

dataSet=myTools.oneDimension(dataSet)

dataSet=dataSet.astype(numpy.uint8)

dataSet=myTools.cropCenter(dataSet, 80)

dataSet=myTools.augmentData(dataSet, numOfTiles=4, overlap=False, imageWidth=819, imageHeight=819)
	
dataSet=dataSet.astype(numpy.float32)
#plt.show(plt.imshow(dataSet[0][0], cmap=cm.binary))

### MASKS
masks=myTools.loadImages('../../masks', 819, 819, 1)

for x in numpy.nditer(masks, op_flags=['readwrite']):
     if x>0:
             x[...]=1

masks=masks.astype(numpy.float32)

masks=myTools.augmentData(masks, numOfTiles=4, overlap=False, imageWidth=819, imageHeight=819)

masks=masks.astype(numpy.float32)

#plt.show(plt.imshow(masks[0][0], cmap=cm.binary))

#plt.show(plt.imshow(dataSet[0][0], cmap=cm.binary))

### DATASET SPLIT
splitPoint=math.floor(dataSet.shape[0]*0.7)

train=dataSet[0:splitPoint, :, :, :]
test=dataSet[splitPoint+1:dataSet.shape[0], :, :, :]

dataSet=None

#temp=numpy.random.rand(12,1,819,819)
#masks=numpy.random.random_integers(0,1, size=(12,1,819,819)).astype(numpy.int32) 

imgsWidth, imgsHeight =train[0][0].shape

data_size=(None,1,imgsWidth,imgsHeight)

numOfBatches=50
batchSize=math.floor(train.shape[0]/numOfBatches)

myNet=myTools.createNN(data_size, X=train, Y=masks[0:splitPoint, :, :, :], epochs=1, n_batches=numOfBatches, batch_size=batchSize, learning_rate=0.2, w_decay=0.005)


res=myNet(test)

print(myTools.myTestCostFunction(res,masks[splitPoint+1:masks.shape[0], :, :, :]))

print(sklearn.metrics.log_loss(masks[splitPoint+1:masks.shape[0], :, :, :], res))


trainInstance=train[0]
trainInstance=trainInstance.reshape(1, trainInstance.shape[0], trainInstance.shape[1], trainInstance.shape[2])

res=myNet(trainInstance)

plt.show(plt.imshow(train[0][0], cmap=cm.binary))

plt.show(plt.imshow(res[0][0], cmap=cm.binary))


testInstance=test[0]
testInstance=testInstance.reshape(1, testInstance.shape[0], testInstance.shape[1], testInstance.shape[2])

res=myNet(testInstance)

plt.show(plt.imshow(test[0][0], cmap=cm.binary))

plt.show(plt.imshow(res[0][0], cmap=cm.binary))

#sys.exit()



#loss, acc = val_fn(temp, masks)
#test_error = 1 - acc
#print('Test error: %f' % test_error)

#rand=numpy.random.rand(10,1,819,819)
#randMasks=numpy.random.random_integers(0,1, size=(10,1,819,819))
#randMasks=numpy.random.random_sample(size=(41,1,819,819))







