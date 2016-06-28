import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

from skimage import data, img_as_float
from skimage import exposure

import numpy
import theano
import theano.tensor as T

import myTools



def myCostFunction(a, b):
  

    r=np.float32(0.1)

    sensitivity=r*T.sum(((a - b)**2)*a)/T.sum(a)
    
    specificity=(1-r)*T.sum(((a - b)**2)*(1-a))/T.sum(1-a)
	
    return sensitivity+specificity


gt=numpy.zeros((2,2))
gt[0,0]=1
gt[1,0]=1

s=numpy.zeros((2,2))
s[0,0]=1
s[1,1]=1

print(gt)
print(s)

print(squared_error(s,gt))







