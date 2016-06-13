import myTools

import theano
import numpy
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


dataSet=myTools.loadImages('/home/athanasiostsiaras/Downloads/images', 1024, 1024, 4)

dataSet=myTools.oneDimension(dataSet)
	
#plt.show(plt.imshow(dataSet[0][0], cmap=cm.binary))

masks=myTools.loadImages('/home/athanasiostsiaras/Downloads/masks', 819, 819, 1)

#plt.show(plt.imshow(masks[0][0], cmap=cm.binary))

cropy=(1024-819)/2
cropx=(1024-819)/2

temp=dataSet[:,:,cropx:1024-cropx-1,cropy:1024-cropy-1]

myTools.myConvNet(temp,masks)


for x in numpy.nditer(masks, op_flags=['readwrite']):
     if x>0:
             x[...]=1


data_size=(None,1,819,819)

input_var = T.tensor4('input')
target_var = T.tensor4('targets')

net = {}

#Input layer:
net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)

#Convolution + Pooling
net['conv1'] = lasagne.layers.Conv2DLayer(net['data'], num_filters=10, filter_size=5)
net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'], pool_size=2)
net['unpool1']=lasagne.layers.InverseLayer(net['pool1'], layer=net['pool1'])
net['deconv1']=lasagne.layers.InverseLayer(net['unpool1'], layer=net['conv1'])


lr = 1e-2
weight_decay = 1e-5

#Loss function: mean cross-entropy
prediction = lasagne.layers.get_output(net['deconv1'])
loss = lasagne.objectives.squared_error(prediction, target_var)
loss = loss.mean()

#Also add weight decay to the cost function
weightsl2 = lasagne.regularization.regularize_network_params(net['deconv1'], lasagne.regularization.l2)
loss += weight_decay * weightsl2

params = lasagne.layers.get_all_params(net['deconv1'], trainable=True)
updates = lasagne.updates.sgd(loss, params, learning_rate=lr)

train_fn = theano.function([input_var, target_var], loss, updates=updates)

test_prediction = lasagne.layers.get_output(net['deconv1'], deterministic=True)
test_loss = lasagne.objectives.squared_error(test_prediction,target_var)
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
get_preds = theano.function([input_var], test_prediction)

n_examples = temp.shape[0]
n_batches = n_examples / batch_size

epochs=2
n_batches=2
batch_size=10

for epoch in xrange(epochs):
    for batch in xrange(n_batches):
        x_batch = temp[batch*batch_size: (batch+1) * batch_size]
        y_batch = masks[batch*batch_size: (batch+1) * batch_size]
        
        train_fn(x_batch, y_batch)


