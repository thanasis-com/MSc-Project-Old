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

#import transposeConv


dataSet=myTools.loadImages('/home/athanasiostsiaras/Downloads/images', 1024, 1024, 4)

dataSet=myTools.oneDimension(dataSet)
	
#plt.show(plt.imshow(dataSet[0][0], cmap=cm.binary))

masks=myTools.loadImages('/home/athanasiostsiaras/Downloads/masks', 819, 819, 1)

#plt.show(plt.imshow(masks[0][0], cmap=cm.binary))

cropy=(1024-819)/2
cropx=(1024-819)/2

temp=dataSet[0:20,:,cropx:1024-cropx-1,cropy:1024-cropy-1]
test=dataSet[21:40,:,cropx:1024-cropx-1,cropy:1024-cropy-1]

#myTools.myConvNet(temp,masks)


for x in numpy.nditer(masks, op_flags=['readwrite']):
     if x>0:
             x[...]=1

plt.show(plt.imshow(masks[0][0], cmap=cm.binary))

#masks=masks.astype(numpy.int32)

#temp=numpy.random.rand(12,1,819,819)
#masks=numpy.random.random_integers(0,1, size=(12,1,819,819)).astype(numpy.int32) 

data_size=(None,1,819,819)

input_var = T.tensor4('input')
target_var = T.tensor4('targets')	

net = {}

#Input layer:
net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)

#Convolution + Pooling
net['conv1'] = lasagne.layers.Conv2DLayer(net['data'], num_filters=10, filter_size=5)
net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'], pool_size=2)
net['conv2'] = lasagne.layers.Conv2DLayer(net['pool1'], num_filters=10, filter_size=5)
net['pool2'] = lasagne.layers.Pool2DLayer(net['conv2'], pool_size=2)
net['unpool2']=lasagne.layers.InverseLayer(net['pool2'], net['pool2'])
net['deconv2']=myClasses.Deconv2DLayer(net['unpool2'], num_filters=10, filter_size=5)
net['unpool1']=lasagne.layers.InverseLayer(net['deconv2'], net['pool1'])
net['deconv1']=myClasses.Deconv2DLayer(net['unpool1'], num_filters=1, filter_size=5, nonlinearity=lasagne.nonlinearities.sigmoid)
#net['conv2']=lasagne.layers.Conv2DLayer(net['pool1'], num_filters=1, filter_size=5)
#net['shape']=lasagne.layers.ReshapeLayer(net['conv2'], shape= ([0], -1))

print(lasagne.layers.get_output_shape(net['data']))
print(lasagne.layers.get_output_shape(net['conv1']))
print(lasagne.layers.get_output_shape(net['pool1']))
print(lasagne.layers.get_output_shape(net['conv2']))
print(lasagne.layers.get_output_shape(net['pool2']))
print(lasagne.layers.get_output_shape(net['unpool2']))
print(lasagne.layers.get_output_shape(net['deconv2']))
print(lasagne.layers.get_output_shape(net['unpool1']))
print(lasagne.layers.get_output_shape(net['deconv1']))
#print(lasagne.layers.get_output_shape(net['shape']))

print(lasagne.layers.count_params(net['deconv1']))

sys.exit()

lr = 0.1
weight_decay = 1e-5

#Loss function: mean cross-entropy
prediction = lasagne.layers.get_output(net['deconv1'])
loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
loss = loss.mean()

#Also add weight decay to the cost function
weightsl2 = lasagne.regularization.regularize_network_params(net['deconv1'], lasagne.regularization.l2)
loss += weight_decay * weightsl2

params = lasagne.layers.get_all_params(net['deconv1'], trainable=True)
updates = lasagne.updates.sgd(loss, params, learning_rate=lr)

train_fn = theano.function([input_var, target_var], loss, updates=updates)

test_prediction = lasagne.layers.get_output(net['deconv1'], deterministic=True)
test_loss = lasagne.objectives.binary_crossentropy(test_prediction,target_var)
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
get_preds = theano.function([input_var], test_prediction)

n_examples = temp.shape[0]
#n_batches = n_examples / batch_size

epochs=200
n_batches=20
batch_size=1

start_time = time.time()

cost_history=[]
batch_cost_history=[]

for epoch in xrange(epochs):
	epoch_time_start=time.time()
    	for batch in xrange(n_batches):
        	x_batch = temp[batch*batch_size: (batch+1) * batch_size]
        	y_batch = masks[batch*batch_size: (batch+1) * batch_size]
        
        	this_cost = train_fn(x_batch, y_batch)
	
		batch_cost_history.append(this_cost)

    	epoch_cost = np.mean(batch_cost_history)
    	cost_history.append(epoch_cost)
    	epoch_time_end = time.time()
    	print('Epoch %d/%d, train error: %f. Elapsed time: %.2f seconds' % (epoch+1, epochs, epoch_cost, epoch_time_end-epoch_time_start))

end_time = time.time()
print('Training completed in %.2f seconds.' % (end_time - start_time))


res=get_preds(test)

#print(res[0][0])

#for x in numpy.nditer(res, op_flags=['readwrite']):
#     if x>0.5:
#             x[...]=1
#     else:
#	      x[...]=0

plt.show(plt.imshow(test[0][0], cmap=cm.binary))

plt.show(plt.imshow(res[0][0], cmap=cm.binary))



#loss, acc = val_fn(temp, masks)
#test_error = 1 - acc
#print('Test error: %f' % test_error)

#rand=numpy.random.rand(10,1,819,819)
#randMasks=numpy.random.random_integers(0,1, size=(10,1,819,819))
#randMasks=numpy.random.random_sample(size=(41,1,819,819))






