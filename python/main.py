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
import sklearn
from lasagne.layers import InputLayer, Conv2DLayer, DenseLayer, MaxPool2DLayer, InverseLayer


argLR=float(sys.argv[1])

argWD=float(sys.argv[2])

print('Learning rate: %f' % (argLR))
print('Weight decay: %f' % (argWD))

### DATASET
dataSet=myTools.loadImages('../../images', 1024, 1024, 4)

dataSet=myTools.oneDimension(dataSet)

dataSet=dataSet.astype(numpy.uint8)

dataSet=myTools.cropCenter(dataSet, 80)

dataSet=myTools.augmentData(dataSet, numOfTiles=4, overlap=False, imageWidth=819, imageHeight=819)
	
dataSet=dataSet.astype(numpy.float32)


### MASKS
masks=myTools.loadImages('../../masks', 819, 819, 1)

for x in numpy.nditer(masks, op_flags=['readwrite']):
     if x>0:
             x[...]=1

masks=masks.astype(numpy.float32)

masks=myTools.augmentData(masks, numOfTiles=4, overlap=False, imageWidth=819, imageHeight=819)

masks=masks.astype(numpy.float32)


### DATASET SPLIT
splitPoint=math.floor(dataSet.shape[0]*0.7)

train=dataSet[0:splitPoint, :, :, :]
test=dataSet[splitPoint+1:dataSet.shape[0], :, :, :]

dataSet=None


imgsWidth, imgsHeight =train[0][0].shape

data_size=(None,1,imgsWidth,imgsHeight)

numOfBatches=50
batchSize=math.floor(train.shape[0]/numOfBatches)


myNet=myTools.createNN(data_size, X=train, Y=masks[0:splitPoint, :, :, :], epochs=1, n_batches=numOfBatches, batch_size=batchSize, learning_rate=argLR, w_decay=argWD)


res=myNet(test)

print('Total cost on test set: %f' % (myTools.myTestCostFunction(res,masks[splitPoint+1:masks.shape[0], :, :, :])))








