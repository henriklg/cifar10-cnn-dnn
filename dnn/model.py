
"""Define the dense neural network model"""

import numpy as np
from scipy.stats import truncnorm
import sys


def one_hot(Y, num_classes):
    """Perform one-hot encoding on input Y.

    It is assumed that Y is a 1D numpy array of length m_b (batch_size) with integer values in
    range [0, num_classes-1]. The encoded matrix Y_tilde will be a [num_classes, m_b] shaped matrix
    with values

                   | 1,  if Y[i] = j
    Y_tilde[i,j] = |
                   | 0,  else
    """
    m = len(Y)
    Y_tilde = np.zeros((num_classes, m))
    Y_tilde[Y, np.arange(m)] = 1
    return Y_tilde


def initialization(conf):
    """Initialize the parameters of the network.

    Args:
        layer_dimensions: A list of length L+1 with the number of nodes in each layer, including
                          the input layer, all hidden layers, and the output layer.
    Returns:
        params: A dictionary with initialized parameters for all parameters (weights and biases) in
                the network.
    """
    
    params = {}
    layer = 1
    prev_lay = conf['layer_dimensions'][0]
    for nodes in conf['layer_dimensions'][1:]:
        cur_lay = nodes
        params["W_"+str(layer)] = np.random.normal(0,2/(prev_lay),size=[prev_lay,cur_lay])
        params["b_"+str(layer)] = np.zeros((nodes,1))
        prev_lay = cur_lay
        layer += 1
        
    return params


def activation(Z, activation_function):
    """Compute a non-linear activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    
    if activation_function == 'relu':
        return np.maximum(0,Z)
    else:
        print("Error: Unimplemented activation function: {}", activation_function)
        return None


def softmax(Z):
    """Compute and return the softmax of the input.

    To improve numerical stability, we do the following

    1: Subtract Z from max(Z) in the exponentials
    2: Take the logarithm of the whole softmax, and then take the exponential of that in the end

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    
    # follow this for improving numerical stability
    # https://www.uio.no/studier/emner/matnat/ifi/IN5400/v19/material/week3/in5400_lectures_week03_slides.pdf
    expo = np.exp(Z)
    soft = expo/np.sum(expo,axis=0)
    
    return soft


def forward(conf, X_batch, params, is_training):
    """One forward step.

    Args:
        conf: Configuration dictionary.
        X_batch: float numpy array with shape [n^[0], batch_size]. Input image batch.
        params: python dict with weight and bias parameters for each layer.
        is_training: Boolean to indicate if we are training or not. This function can namely be
                     used for inference only, in which case we do not need to store the features
                     values.

    Returns:
        Y_proposed: float numpy array with shape [n^[L], batch_size]. The output predictions of the
                    network, where n^[L] is the number of prediction classes. For each input i in
                    the batch, Y_proposed[c, i] gives the probability that input i belongs to class
                    c.
        features: Dictionary with
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
               We cache them in order to use them when computing gradients in the backpropagation.
    """
    
    """
    # features
    features = {}
    Y_proposed = None
    L = len(conf.get("layer_dimensions"))-1
    
    if (is_training):
        # input layer
        W = params['W_1']
        b = params['b_1']
        A = X_batch
        
        features['Z_1'] = W.T.dot(A) + b #np.outer(params['b_1'], np.ones(np.shape(A)[1]))
        features['A_1'] = activation(features['Z_1'], conf['activation_function'])
        
        # iterate over layers l.. L-1
        for l in range(2,L-1):
            W = params.get("W_"+str(l))
            b = params.get("b_"+str(l))
            features["Z_"+str(l)] = W.T.dot(features['A_'+str(l)]) + b
            features["A_"+str(l)] = activation(features['Z_'+str(l)], conf['activation_function'])
        
        features["Z_"+str(L)] = params['W_'+str(L)].T.dot(features['A_'+str(L-1)]) + params['b_'+str(L)]
        features["A_"+str(L)] = softmax(features['Z_'+str(L)])  # riktig?

    
        # Y_proposed
        Y_proposed = softmax(features['Z_'+str(L)])
    
    return Y_proposed, features
    """
    
    L = len(params)//2
    features = {}
    features['A_0'] = X_batch
    for l in range(L):
        W = params['W_%d'%(l+1)]
        b = params['b_%d'%(l+1)]
        A = features['A_%d'%l]
        features['Z_%d'%(l+1)] = np.dot(W.T,A)+b
        Z = features['Z_%d'%(l+1)]
        features['A_%d'%(l+1)] = activation(Z,conf['activation_function'])
    Y_proposed = softmax(Z)
    #print(Y_proposed)
    if (not(is_training)):
        features = None
    return Y_proposed, features



def cross_entropy_cost(Y_proposed, Y_reference):
    """Compute the cross entropy cost function.

    Args:
        Y_proposed: numpy array of floats with shape [n_y, m].
        Y_reference: numpy array of floats with shape [n_y, m]. Collection of one-hot encoded
                     true input labels

    Returns:
        cost: Scalar float: 1/m * sum_i^m sum_j^n y_reference_ij log y_proposed_ij
        num_correct: Scalar integer
    """
    n,m = np.shape(Y_proposed)
    cost = -1/m * np.sum(Y_reference*np.log(Y_proposed))
    Y_max = np.zeros((n,m))
    Y_max[np.where(Y_proposed==np.max(Y_proposed,axis=0))] = 1
    num_correct = np.sum(Y_max*Y_reference)

    return cost, num_correct


def activation_derivative(Z, activation_function):
    """Compute the gradient of the activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.4 a)
    if activation_function == 'relu':
        d_r = np.zeros(Z.shape)
        d_r[np.where(Z >= 0)] = 1
        return d_r
    else:
        print("Error: Unimplemented derivative of activation function: {}", activation_function)
        return None


def backward(conf, Y_proposed, Y_reference, params, features):
    """Update parameters using backpropagation algorithm.

    Args:
        conf: Configuration dictionary.
        Y_proposed: numpy array of floats with shape [n_y, m].
        features: Dictionary with matrices from the forward propagation. Contains
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
        params: Dictionary with values of the trainable parameters.
                - the weights W^[l] for l in [1, L].
                - the biases b^[l] for l in [1, L].
    Returns:
        grad_params: Dictionary with matrices that is to be used in the parameter update. Contains
                - the gradient of the weights, grad_W^[l] for l in [1, L].
                - the gradient of the biases grad_b^[l] for l in [1, L].
    """
    
    grad_params = {}
    L = len(params)//2
    n,m = np.shape(Y_proposed)
    J = Y_proposed - Y_reference
    grad_params['grad_W_'+str(L)] = (1/m)*np.dot(features['A_'+str(L-1)],np.transpose(J))
    grad_params['grad_b_'+str(L)] = (1/m)*np.dot(J,np.ones((m,1)))
    for i in range(1,L):
        GoZ = activation_derivative(features['Z_'+str(L-i)],conf['activation_function'])  # g of Z
        dotJW = np.dot(params['W_'+str(L-i+1)],J)
        J = GoZ * dotJW
        grad_params['grad_W_'+str(L-i)] = (1/m)*np.dot(features['A_'+str(L-i-1)],np.transpose(J))
        grad_params['grad_b_'+str(L-i)] = (1/m)*np.dot(J,np.ones((m,1)))
    
    return grad_params


def gradient_descent_update(conf, params, grad_params):
    """Update the parameters in params according to the gradient descent update routine.

    Args:
        conf: Configuration dictionary
        params: Parameter dictionary with W and b for all layers
        grad_params: Parameter dictionary with b gradients, and W gradients for all
                     layers.
    Returns:
        params: Updated parameter dictionary.
    """
    
    updated_params = {}
    L = len(params)//2
    for i in range(1,L+1):
        updated_params['W_'+str(i)] = params['W_'+str(i)] - conf['learning_rate']*grad_params['grad_W_'+str(i)]
        updated_params['b_'+str(i)] = params['b_'+str(i)] - conf['learning_rate']*grad_params['grad_b_'+str(i)]
    return updated_params



