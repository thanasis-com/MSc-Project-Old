from collections import Iterable
import numpy
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import gpu_contiguous
#from theano import theano_extensions

#from .. import init
from lasagne import nonlinearities
#from utils import as_tuple
from lasagne.layers import get_output, get_output_shape
#from theano_extensions import conv

#from .base import Layer


class TransposeConv2DLayer(BaseConvLayer):
    """An upsampling Layer that transposes a convolution.
    This layer upsamples its input using the transpose of a convolution,
    also known as fractional convolution in some contexts.
    This is the upsample layer used in [1]_. For a thorough description
    of the operation refer to [2]_.
    Parameters
    ----------
    input_shape : [None/int/Constant] * 2 + [Tensor/int/Constant] * 2
            The shape of the input (upsampled) parameter.
            A tuple/list of len 4, with the first two dimensions
            being None or int or Constant and the last two dimensions being
            Tensor or int or Constant. If None defaults to Lasagne's shape
            inference on the `incoming` input argument, which will fail
            in case `batches` or `channels` are not fixed-sized.
    References
    ----------
    .. [1] Francesco Visin et Al. (2015):
           ReSeg: A Recurrent Neural Network for Object Segmentation
    .. [2] Vincent Dumoulin and Francesco Visin (2016):
           A guide to convolution arithmetic for deep learning
    Notes
    -----
    Expects the input to be in format: batches, channels, rows, cols
    """
    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
                 untie_biases=False, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=None, flip_filters=True,
                 in_shape=None, **kwargs):
        if in_shape is None:
            tensor_shape = get_output(incoming).shape
            in_shape = get_output_shape(
                incoming)[:2] + (tensor_shape[2],) + (tensor_shape[3],)
        self.in_shape = in_shape
        super(TransposeConv2DLayer, self).__init__(
            incoming, num_filters, filter_size, stride, pad, untie_biases, W,
            b, nonlinearity, flip_filters, **kwargs)

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.
        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.in_shape[1]
        return (num_input_channels, self.num_filters) + self.filter_size

    def get_output_shape_for(self, input_shape):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) + tuple(compute_tconv_out_size(
            input_shape[2:], self.filter_size, self.stride, pad)))

    def get_output_for(self, input, **kwargs):
        conved = self.convolve(input, **kwargs)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            # activation = conved + T.shape_padleft(self.b, 1)
            activation = conved + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = conved + self.b.dimshuffle(('x', 0) + ('x',) * self.n)

        return self.nonlinearity(activation)

    def convolve(self, input, **kwargs):
        """
        Returns the symbolical transposed convolution of `input` with
        ``self.W``, producing an output of shape ``self.output_shape``.
        Parameters
        ----------
        input : Theano tensor
            The input minibatch to convolve on
        **kwargs
            Any additional keyword arguments from :meth:`get_output_for`
        Returns
        -------
        Theano tensor
            `input` convolved according to the configuration of this layer,
            without any bias or nonlinearity applied.
        """
        filters = gpu_contiguous(self.W)
        input = gpu_contiguous(input)
        kshp = [None if isinstance(el, T.TensorVariable) else
                el for el in self.get_W_shape()]
        in_shape = self.in_shape
        out_shape = compute_tconv_out_size(in_shape[2:], self.filter_size,
                                           self.stride, self.pad)
        for el in out_shape:
            if isinstance(el, T.TensorVariable):
                el = None
        out_shape = [in_shape[0]] + [self.num_filters] + list(out_shape)

        return T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
            output_grad=input,
            filters=filters,
            input_shape=out_shape,
            filter_shape=kshp,
            border_mode=self.pad,
            subsample=self.stride,
            filter_flip=self.flip_filters)


def compute_tconv_out_size(input_size, filter_size, stride, pad):
    """Computes the length of the output of a transposed convolution
    Parameters
    ----------
    input_size : int, Iterable or Theano tensor
        The size of the input of the transposed convolution
    filter_size : int, Iterable or Theano tensor
        The size of the filter
    stride : int, Iterable or Theano tensor
        The stride of the transposed convolution
    pad : int, Iterable, Theano tensor or string
        The padding of the transposed convolution
    """
    if input_size is None:
        return None
    input_size = numpy.array(input_size)
    filter_size = numpy.array(filter_size)
    stride = numpy.array(stride)

    if isinstance(pad, (int, Iterable)) and not isinstance(pad, str):
        pad = numpy.array(pad)  # to deal with iterables in one line
        output_size = (input_size - 1) * stride + filter_size - 2*pad
    elif pad == 'full':
        output_size = input_size * stride - filter_size - stride + 2
    elif pad == 'valid':
        output_size = (input_size - 1) * stride + filter_size
    elif pad == 'same':
        output_size = input_size
    return output_size
