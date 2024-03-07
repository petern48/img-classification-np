import numpy as np
from nndl.layers import *
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

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  H_new = 1 + (H + 2 * pad - HH) // stride
  W_new = 1 + (W + 2 * pad - WW) // stride
  out = np.zeros((N, F, H_new, W_new))

  if pad > 0:
    xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode="constant")
  else:
    xpad = x
  # for each filter
  for f in range(F):
    # convolve down
    for h in range(0, H, stride):
      # convolve right
      for ww in range(0, W, stride):
        filter = w[f, :, :, :][None, :, :, :]  # (1, C, HH, WW)
        window = xpad[:, :, h: h + HH, ww: ww + WW]  # (N, C, HH, WW)
        val = np.sum((filter * window).reshape(N, -1), axis=1) + b[f]  # (N,)
        out[:, f, h // stride, ww // stride] = val
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  F, HH, WW = num_filts, f_height, f_width
  db = np.sum(dout, axis=(0, 2, 3))  # (F) sum all except axis 1
  dw = np.zeros_like(w)  # (F, C, HH, WW)
  dxpad = np.zeros_like(xpad)  # (N, C, H + 2pad, W + 2pad)
  # dout (N, F, out_H, out_W)
  for f in range(F):
    for h in range(0, out_height):
      for w2 in range(0, out_width):
        dout_ele = dout[:, f, h, w2][:, None, None, None]  # (N,1,1,1)
        hx, wx = h * stride, w2 * stride
        window_slice = np.index_exp[:, :, hx: hx + HH, wx: wx + WW]  # (N, C, HH, WW)
        # dw
        dw[f,:,:,:] += np.sum(xpad[window_slice] * dout_ele, axis=0)  # (C, HH, WW)
        # dx
        dxpad[window_slice] += w[f, :, :, :] * dout_ele
  # unpad
  dx = dxpad[:, :, pad: -pad, pad: -pad]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  N, C, H, W = x.shape
  ph, pw, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  out = np.zeros((N, C, (H - ph) // stride + 1, (W - pw) // stride + 1))

  for h in range(0, H, stride):
    for w in range(0, W, stride):
      window = x[:, :, h: h+ph, w: w+pw]
      out[:, :, h // stride, w // stride] = np.max(window, axis=(2, 3))  # (N, C)
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  ph, pw = pool_height, pool_width
  N, C, out_H, out_W = dout.shape
  dx = np.zeros_like(x)
  for h in range(out_H):
    for w in range(out_W):
      hx, wx = h * stride, w * stride
      # for n in range(N):
      #   for c in range(C):
      dout_ele = dout[:, :, h, w][:, :, None, None]  # (N, C, 1, 1)
      window = x[:, :, hx: hx + ph, wx: wx + pw]  # (N, C, ph, pw)
      max_vals = np.max(window, axis=(2,3))  # (N, C)
      max_vals = max_vals[:,:,None].repeat(ph, axis=2)[:,:,:, None].repeat(pw, axis=3)  # (N, C, ph, pw)
      mask = np.where(window == max_vals, 1, 0)  # (N, C, ph, ph)
      dx[:, :, hx: hx + ph, wx: wx + pw] += dout_ele * mask  # (N, C, ph, pw)
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = x.shape
  x = x.transpose(0,2,3,1)  # (N,C,H,W) -> (N, H, W, C)
  x = x.reshape(-1, C)
  out, cache = batchnorm_forward(x, gamma, beta, bn_param)
  out = out.reshape(N, H, W, C)  # uncollapse
  out = out.transpose(0, 3, 1, 2)
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = dout.shape
  dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)
  dx, dgamma, dbeta = batchnorm_backward(dout, cache)
  dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta