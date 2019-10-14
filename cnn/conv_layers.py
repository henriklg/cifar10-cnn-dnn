"""Implementation of convolution forward and backward pass"""

import numpy as np

def conv_layer_forward(input_layer, weight, bias, pad_size=1, stride=1):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of M data points, each with C channels, height H and
    width W. We convolve each input with C_o different filters, where each filter
    spans all C_i channels and has height H_w and width W_w.

    Args:
        input_layer: The input layer with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        output_layer: The output layer with shape (batch_size, num_filters, height_y, width_y)
    """

    (batch_size, channels_x, height_x, width_x) = input_layer.shape
    (num_filters, channels_w, height_w, width_w) = weight.shape

    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")
    
    output_dimension = 1+(height_x+(2*pad_size)-height_w)//stride
    output_layer = np.zeros((batch_size,num_filters,output_dimension,output_dimension))
    
    for batch in range(batch_size):
        for curr_f in range(num_filters):
            for channel in range(channels_w):
                image_pad = np.pad(input_layer[batch,channel,:,:],pad_size,mode='constant')
                kernel = weight[curr_f,channel,:,:]
                for x in range(0,height_x,stride):
                    for y in range(0,width_x,stride):
                        image = image_pad[x:x+width_w,y:y+height_w]
                        output_layer[batch, curr_f, x//stride, y//stride] += np.sum(kernel * image)
            output_layer[batch,curr_f,:,:] += bias[curr_f]
    return output_layer
    


def conv_layer_backward(output_layer_gradient, input_layer, weight, bias, pad_size=1):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Args:
        output_layer_gradient: Gradient of the loss L wrt the next layer y, with shape
            (batch_size, num_filters, height_y, width_y)
        input_layer: Input layer x with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        input_layer_gradient: Gradient of the loss L with respect to the input layer x
        weight_gradient: Gradient of the loss L with respect to the filters w
        bias_gradient: Gradient of the loss L with respect to the biases b
    """
    
    input_layer_gradient = np.zeros(np.shape(input_layer))
    weight_gradient = np.zeros(np.shape(weight))
    bias_gradient =  np.zeros(np.shape(bias))

    batch_size, channels_y, height_y, width_y = output_layer_gradient.shape
    batch_size, channels_x, height_x, width_x = input_layer.shape
    num_filters, channels_w, height_w, width_w = weight.shape

    assert num_filters == channels_y, (
        "The number of filters must be the same as the number of output layer channels")
    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")
    
    pad_array = ((0,0),(0,0),(pad_size,pad_size),(pad_size,pad_size))
    output_pad = np.pad(output_layer_gradient,(pad_array),mode = 'constant')
    input_pad = np.pad(input_layer,(pad_array),mode = 'constant')
    weight_rot = np.rot90(weight,2,(2,3))
    
    for batch in range(batch_size):
        for curr_f in range(num_filters):
            # bias gradient
            bias_gradient[curr_f] += np.sum(output_layer_gradient[batch,curr_f,:,:])
            for channel in range(channels_x):
                
                # input layer gradient
                for x in range(height_x):
                    for y in range(width_x):
                        image = output_pad[batch,curr_f,x:x+width_w,y:y+height_w]
                        input_layer_gradient[batch, channel, x, y] += np.sum(image*weight_rot[curr_f,channel])
                
                # weight gradient
                for x in range(height_w):
                    for y in range(width_w):
                        image = input_pad[batch,channel,x:width_x+x,y:height_x+y]
                        weight_gradient[curr_f,channel,x,y] += np.sum(output_layer_gradient[batch,curr_f] * image)
                        
    return input_layer_gradient, weight_gradient, bias_gradient




def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad
