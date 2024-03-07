import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from utils.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #
    C, H, W = input_dim
    num_hidden_layers = 2
    self.params["W1"] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params["b1"] = np.zeros(num_filters)

    conv_out_dim = (num_filters * (H // 2) * (W // 2))
    self.params["W2"] = weight_scale * np.random.randn(conv_out_dim, hidden_dim)
    self.params["b2"] = np.zeros(hidden_dim)

    self.params["W3"] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params["b3"] = np.zeros(num_classes)

    if self.use_batchnorm:
      self.bn_params = []
      for i in range(1, num_hidden_layers + 1):
        out_dim = num_filters if i == 1 else hidden_dim
        self.params[f"gamma{i}"] = np.ones(out_dim)
        self.params[f"beta{i}"] = np.zeros(out_dim)
        self.bn_params.append({"mode": "train"})  # running mean, var, momentum
    # num_hidden_layers = 2
    # for i in range(1, num_hidden_layers + 1):  # all except 3rd layer
    #   out_dim = num_filters if i == 1 else hidden_dim
    #   if i == 1:  # conv layer
    #     self.params[f"W{i}"] = weight_scale * np.random.randn(out_dim, C, filter_size, filter_size)
    #     self.params[f"b{i}"] = np.zeros(num_filters)
    #   else:  # affine
    #     # assume H and W stay same coming out of the conv layer, then // 2 for max pool
    #     conv_out_dim = (num_filters * (H // 2) * (W // 2))  # collapse conv out dims (F, H, W)
    #     self.params[f"W{i}"] = weight_scale * np.random.randn(conv_out_dim, out_dim)
    #     self.params[f"b{i}"] = np.zeros(hidden_dim)

      # self.params[f"b{i}"] = np.zeros(out_dim)

    # final layer: affine 2
    # self.params["W3"] = weight_scale * np.random.randn(hidden_dim, num_classes)
    # self.params["b3"] = np.zeros(num_classes)
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #
    out = X
    cache = {}
    # Layer 1
    out, cache["conv1"] = conv_forward_fast(out, W1, b1, conv_param)
    if self.use_batchnorm:
      gamma1, beta1 = self.params["gamma1"], self.params["beta1"]
      out, cache["bn1"] = spatial_batchnorm_forward(out, gamma1, beta1, self.bn_params[0])
    out, cache["relu1"] = relu_forward(out)
    out, cache["pool2"] = max_pool_forward_fast(out, pool_param)
    # Layer 2
    out, cache["fc2"] = affine_forward(out, W2, b2)
    if self.use_batchnorm:
      gamma2, beta2 = self.params["gamma2"], self.params["beta2"]
      out, cache["bn2"] = batchnorm_forward(out, gamma2, beta2, self.bn_params[1])
    out, cache["relu2"] = relu_forward(out)
    # Layer 3
    scores, cache["fc3"] = affine_forward(out, W3, b3)
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #

    loss, dout  = softmax_loss(scores, y)
    dfc3, grads["W3"], grads["b3"] = affine_backward(dout, cache["fc3"])
    dre2 = relu_backward(dfc3, cache["relu2"])
    if self.use_batchnorm:
      dre2, grads["gamma2"], grads["beta2"] = batchnorm_backward(dre2, cache["bn2"])
    dfc2, grads["W2"], grads["b2"] = affine_backward(dre2, cache["fc2"])
    dpo = max_pool_backward_fast(dfc2, cache["pool2"])
    dre1 = relu_backward(dpo, cache["relu1"])
    if self.use_batchnorm:
      dre1, grads["gamma1"], grads["beta1"] = spatial_batchnorm_backward(dre1, cache["bn1"])
    _, grads["W1"], grads["b1"] = conv_backward_fast(dre1, cache["conv1"])

    # Update loss and grads for reg term
    reg_loss = 0.0
    for i in range(1, 3+1):
      reg_loss += np.sum(self.params[f"W{i}"] ** 2)
      grads[f"W{i}"] += self.reg * self.params[f"W{i}"]
    reg_loss *= 0.5 * self.reg
    loss += reg_loss
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
  
  
pass
