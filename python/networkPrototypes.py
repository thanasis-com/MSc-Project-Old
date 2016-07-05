
### MY INITIAL ARCHITECTURE ###
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


### MY INITIAL ARCHITECTURE WITH BATCH NORMALISATION ON EVERY CONV-DECONV LEAYER ###
#Input layer:
	net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)

	#the rest of the network structure
	net['conv1'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['data'], num_filters=10, filter_size=5))
	net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'], pool_size=2)
	net['conv2'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool1'], num_filters=10, filter_size=5))
	net['pool2'] = lasagne.layers.Pool2DLayer(net['conv2'], pool_size=2)
	net['unpool2']= lasagne.layers.InverseLayer(net['pool2'], net['pool2'])
	net['deconv2']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool2'], num_filters=10, filter_size=5))
	net['unpool1']= lasagne.layers.InverseLayer(net['deconv2'], net['pool1'])
	net['output']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool1'], num_filters=1, filter_size=5, nonlinearity=lasagne.nonlinearities.sigmoid))


### SAME AS ABOVE BUT WITH MORE CONV AND DECONV LAYERS ###
#Input layer:
	net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)

	#the rest of the network structure
	net['conv1'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['data'], num_filters=10, filter_size=5))
	net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'], pool_size=2)
	net['conv2'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool1'], num_filters=10, filter_size=5))
	net['pool2'] = lasagne.layers.Pool2DLayer(net['conv2'], pool_size=2)
	net['conv3'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool2'], num_filters=10, filter_size=5))
	net['pool3'] = lasagne.layers.Pool2DLayer(net['conv3'], pool_size=2)
	net['unpool3']= lasagne.layers.InverseLayer(net['pool3'], net['pool3'])
	net['deconv3']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool3'], num_filters=10, filter_size=5))
	net['unpool2']= lasagne.layers.InverseLayer(net['deconv3'], net['pool2'])
	net['deconv2']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool2'], num_filters=10, filter_size=5))
	net['unpool1']= lasagne.layers.InverseLayer(net['deconv2'], net['pool1'])
	net['output']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool1'], num_filters=1, filter_size=5, nonlinearity=lasagne.nonlinearities.sigmoid))


### SAME AS ABOVE BUT WITH MORE CONV AND DECONV LAYERS ###
#Input layer:
	net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)

	#the rest of the network structure
	net['conv1'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['data'], num_filters=10, filter_size=5))
	net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'], pool_size=2)
	net['conv2'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool1'], num_filters=10, filter_size=5))
	net['pool2'] = lasagne.layers.Pool2DLayer(net['conv2'], pool_size=2)
	net['conv3'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool2'], num_filters=10, filter_size=5))
	net['pool3'] = lasagne.layers.Pool2DLayer(net['conv3'], pool_size=2)
	net['conv4'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool3'], num_filters=10, filter_size=5))
	net['pool4'] = lasagne.layers.Pool2DLayer(net['conv4'], pool_size=2)
	net['unpool4']= lasagne.layers.InverseLayer(net['pool4'], net['pool4'])
	net['deconv4']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool4'], num_filters=10, filter_size=5))
	net['unpool3']= lasagne.layers.InverseLayer(net['deconv4'], net['pool3'])
	net['deconv3']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool3'], num_filters=10, filter_size=5))
	net['unpool2']= lasagne.layers.InverseLayer(net['deconv3'], net['pool2'])
	net['deconv2']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool2'], num_filters=10, filter_size=5))
	net['unpool1']= lasagne.layers.InverseLayer(net['deconv2'], net['pool1'])
	net['output']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool1'], num_filters=1, filter_size=5, nonlinearity=lasagne.nonlinearities.sigmoid))



### A UNET-LIKE ARCHITECTURE ###
#Input layer:
	net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)

	#the rest of the network structure
	net['conv1'] = lasagne.layers.Conv2DLayer(net['data'], num_filters=10, filter_size=5)
	net['conv2'] = lasagne.layers.Conv2DLayer(net['conv1'], num_filters=10, filter_size=5)
	net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'], pool_size=2)
	net['unpool1']=lasagne.layers.InverseLayer(net['pool1'], net['pool1'])
	net['deconv2']=myClasses.Deconv2DLayer(net['unpool1'], num_filters=10, filter_size=5)
	net['output']=myClasses.Deconv2DLayer(net['deconv2'], num_filters=1, filter_size=5, nonlinearity=lasagne.nonlinearities.sigmoid)




#the rest of the network structure
	net['conv1'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['data'], num_filters=4, filter_size=5))
	net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'], pool_size=2)
	net['conv2'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool1'], num_filters=20, filter_size=5))
	net['pool2'] = lasagne.layers.Pool2DLayer(net['conv2'], pool_size=2)
	net['conv3'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool2'], num_filters=20, filter_size=5))
	net['pool3'] = lasagne.layers.Pool2DLayer(net['conv3'], pool_size=2)
	net['conv4'] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(net['pool3'], num_filters=20, filter_size=5))
	#net['pool4'] = lasagne.layers.Pool2DLayer(net['conv4'], pool_size=2)
	#net['unpool4']= lasagne.layers.InverseLayer(net['pool4'], net['pool4'])
	net['deconv4']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['conv4'], num_filters=20, filter_size=5))
	net['unpool3']= lasagne.layers.InverseLayer(net['deconv4'], net['pool3'])
	net['deconv3']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool3'], num_filters=20, filter_size=5))
	net['unpool2']= lasagne.layers.InverseLayer(net['deconv3'], net['pool2'])
	net['deconv2']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool2'], num_filters=20, filter_size=5))
	net['unpool1']= lasagne.layers.InverseLayer(net['deconv2'], net['pool1'])
	net['deconv1']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['unpool1'], num_filters=4, filter_size=5))
	net['output']= lasagne.layers.batch_norm(myClasses.Deconv2DLayer(net['deconv1'], num_filters=1, filter_size=1, nonlinearity=lasagne.nonlinearities.sigmoid))
