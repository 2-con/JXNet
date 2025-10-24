import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import random

# for type checking
from core.standard.functions import Function
from core.standard.initializers import Initializer, Default

from system.config import *

class Layer(ABC):
  """
  Base class for all layers
  
  Considering that layers can have differing operations, make sure the internal components agree with the NetLab API which means
  a functional-programming paradigm and correct capitalization (class methods are always lowercase). Attributes of the layer should stay constant and immutable.
  
  A Layer class is required to have the following:
  - '__init__' : method with any constant object attributes should be defined here
    - Args:
      - Any
    - Returns:
      - None
    
  - 'calibrate' : method that will be called once per layer during compilation to generate a weight matrix as well as constant object attributes
    - Args:
      - fan_in tuple[int,...] : shape of the input to the layer
      - fan_out_shape tuple[int,...] : shape of the output of the layer
    - Returns:
      - dict : weight_matrix (including bias and parametric values) for the whole layer
      - tuple[int,...] : shape of the output of the layer
      
  - 'forward' : method that will be called everytime the layer is called
    - Args:
      - params (dict) : weight_matrix (including bias and parametric values) for the whole layer
      - inputs (jnp.ndarray) : the input to the layer
    - Returns:
      - jnp.ndarray : the output of the layer
      - jnp.ndarray : cached values to be used in the backward pass
      
  - 'backward' : method that will be called before the 'update' method
    - Args:
      - params (dict) : weight_matrix (including bias and parametric values) for the whole layer
      - inputs (jnp.ndarray) : the input to the layer
      - error (jnp.ndarray) : the incoming error for the layer
      - weighted_sums (jnp.ndarray) : cached values from the forward pass
    - Returns:
      - jnp.ndarray : the upstream error to be passed to the next layer
      - dict : weight_matrix (including bias and parametric values) for the whole layer
  
  - 'update' : method that will be called if provided, otherwise NetLab/NetFlash will ignore the layer
    - Args:
      - optimizer (optimizer.forward method) : this could be any callable, but make sure it adheres to the structure provided under lab.optimizer
      - learning_rate (float or jnp.float32) : learning rate
      - layer_params (dict) : weight_matrix (including bias and parametric values) for the whole layer
      - gradients (dict) : error gradients for the whole layer
      - opt_state (dict) : optimizer state for the whole layer
      - **other_parameters (kwargs) : any other optimizer hyperparameters that could be passed in
    - Returns:
      - dict : updated params for the layer
      - dict : updated opt state
  """
  layer_seed = 0
  training_only=False
  
  @abstractmethod
  def __init__(self):
    """
    A Layer class is required to have the following:
    
    - A '__init__' method with any constant object attributes should be defined here
      - Args:
        - Any
      - Returns:
        - None
    """
    pass
  
  @abstractmethod
  def calibrate(self, fan_in:tuple[int,...], fan_out_shape:int) -> tuple[dict, tuple[int,...]]:
    """
    - A 'calibrate' method that will be called once per layer during compilation to generate a weight matrix as well as constant object attributes
      - Args:
        - fan_in tuple[int,...] : shape of the input to the layer
        - fan_out_shape tuple[int,...] : shape of the output of the layer
      - Returns:
        - dict : weight_matrix (including bias and parametric values) for the whole layer
        - tuple[int,...] : shape of the output of the layer
    """
    pass
  
  @abstractmethod
  def forward(self):
    """
    - A 'forward' method that will be called everytime the layer is called
      - Args:
        - params (dict) : weight_matrix (including bias and parametric values) for the whole layer
        - inputs (jnp.ndarray) : the input to the layer
      - Returns:
        - jnp.ndarray : the output of the layer
        - jnp.ndarray : cached values to be used in the backward pass
    """
    pass
  
  @abstractmethod
  def backward(self):
    """
    - A 'backward' method
      - Args:
        - params (dict) : weight_matrix (including bias and parametric values) for the whole layer
        - inputs (jnp.ndarray) : the input to the layer
        - error (jnp.ndarray) : the incoming error for the layer
        - weighted_sums (jnp.ndarray) : cached values from the forward pass
      - Returns:
        - jnp.ndarray : the upstream error to be passed to the next layer
        - dict : weight_matrix (including bias and parametric values) for the whole layer
    """
    pass

class Dense(Layer):
  layer_seed = 0
  training_only=False
  def __init__(self, neurons:int, function:Function, name:str="", initializer:Initializer=Default(), *args, **kwargs):
    """
    Initialize a Dense layer
    -----
      A fully connected layer that accepts and returns 1D arrays with an input shape (input_size,) excluding batch dimension.
      higher dimentions specified will not be flattened automatically and return an error instead.
    -----
    Args
    -----
    - neurons                (int)         : the number of neurons in the layer
    - function               (Function)    : the function function
    - (Optional) name        (str)         : the name of the layer
    - (Optional) initializer (Initializer) : intializer for the weights, defaults to Default
    - (Optional) *args                     : variable length argument list
    - (Optional) **kwargs                  : arbitrary keyword arguments
    """
    self.neuron_amount = neurons
    self.name = name
    self.function = function
    self.initializer = initializer

  def calibrate(self, fan_in_shape:tuple[int,...], fan_out_shape:int) -> tuple[dict, tuple[int,...]]:
    weights = self.initializer(self.layer_seed, (fan_in_shape[0], self.neuron_amount), fan_in_shape[0], fan_out_shape[0])
    biases = jnp.zeros((self.neuron_amount,))
    paremetric_parameters = {
      paramater_name: self.initializer(self.layer_seed, (self.neuron_amount,), fan_in_shape[0], fan_out_shape[0]) for paramater_name in self.function.parameters
    }
    
    return {'weights': weights, 'biases': biases, **paremetric_parameters}, (self.neuron_amount,)

  def forward(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # inputs: (batch, in_features), weights: (in_features, out_features)
    weighted_sums = inputs @ params['weights'] + params['biases']
    activated_output = self.function.forward(weighted_sums, **params)
    return activated_output, weighted_sums
  
  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    # error: (batch, out_features), inputs: (batch, in_features)
    
    all_grads = self.function.backward(error, weighted_sums, **params)
    
    parameter_grads = {}
    for name, grad in all_grads.items():
      
      if name != 'x':
        parameter_grads[name] = grad
    
    grads_z = all_grads['x']
    
    grads_weights = jnp.einsum("bi,bj->ij", inputs, grads_z)  # (in_features, out_features)
    
    grads_biases = jnp.sum(grads_z, axis=0)  # (out_features,)
    upstream_gradient = grads_z @ params['weights'].T  # (batch, in_features)

    param_grads = {
      'weights': grads_weights,
      'biases': grads_biases,
      **parameter_grads
    }

    return upstream_gradient, param_grads

  @staticmethod
  def update(optimizer, learning_rate, layer_params:dict, gradients:jnp.ndarray, opt_state:dict, *args, **kwargs) -> dict:
    updated_params = {}
    new_opt_state = {}
    
    for name, value in layer_params.items():
      updated_params[name], new_opt_state[name] = optimizer.update(
        learning_rate,
        value,
        gradients[name],
        opt_state[name],
        *args,
        **kwargs
      )
    
    return updated_params, new_opt_state

class Localunit(Layer):
  layer_seed = 0
  training_only=False
  def __init__(self, receptive_field:int, function:Function, initializer:Initializer=Default(), name:str="", *args, **kwargs):
    """
    Localunit (Locally Connected Layer)
    -----
      A locally connected layer that accepts and returns 1D arrays with an input shape (input_size,) excluding batch dimension.
      higher dimentions specified will not be flattened automatically and return an error instead.
    -----
    Args
    -----
    - receptive_field        (int)         : the size of the receptive field for each neuron
    - function               (Function)    : the activation function
    - (Optional) name        (str)         : the name of the layer
    - (Optional) initializer (Initializer) : intializer for the weights, defaults to Default
    - (Optional) *args                     : variable length argument list
    - (Optional) **kwargs                  : arbitrary keyword arguments
    """
    self.receptive_field = receptive_field
    self.name = name
    self.function = function
    self.initializer = initializer

  def calibrate(self, fan_in:tuple[int,...], fan_out_shape:tuple[int,...]) -> tuple[dict, tuple[int,...]]:
    
    # generating the slide pattern
    ans = []
    height = (fan_in[0] - self.receptive_field) + 1
    
    if height < 1:
      raise ValueError("Field size must be less than or equal to width.")
    if self.receptive_field < 0 or fan_in[0] < 0:
      raise ValueError("Width or field size must be non-negative.")
    
    # mask builder
    for i in range(height):
      row = [0 for _ in range(fan_in[0])]
      for j in range(fan_in[0]):
        if j+i < fan_in[0] and j+i >= i and j < self.receptive_field:
          row[j+i] = 1
      ans.append(row)

    # permanent localunit weight mask
    self.mask = jnp.array(ans).T
    
    self.input_size = fan_in[0]
    weights = self.initializer(self.layer_seed, (self.mask.shape[0], self.mask.shape[1]), fan_in[0], fan_out_shape[0])
    biases = jnp.zeros((self.mask.shape[1],))
    paremetric_parameters = {
      paramater_name: self.initializer(self.layer_seed, (self.mask.shape[0],), fan_in[0], fan_out_shape[0]) for paramater_name in self.function.parameters
    }
    
    return {'weights': weights, 'biases': biases, **paremetric_parameters}, (self.mask.shape[0],)

  def forward(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # inputs: (batch, in_features), weights: (in_features, out_features)
    weighted_sums = inputs @ (params['weights'] * self.mask) + params['biases']
    activated_output = self.function.forward(weighted_sums, **params)
    return activated_output, weighted_sums

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    # error: (batch, out_features), inputs: (batch, in_features)
    all_grads = self.function.backward(error, weighted_sums, **params)
    
    parameter_grads = {}
    for name, grad in all_grads.items():
      
      if name != 'x':
        parameter_grads[name] = grad
    
    grads_z = all_grads['x']

    grads_weights = jnp.einsum("bi,bj->ij", inputs, grads_z) * self.mask  # (in_features, out_features)
    
    grads_biases = jnp.sum(grads_z, axis=0)  # (out_features,)
    upstream_gradient = grads_z @ params['weights'].T  # (batch, in_features)

    param_grads = {
      'weights': grads_weights,
      'biases': grads_biases,
      **parameter_grads
    }

    return upstream_gradient, param_grads

  @staticmethod
  def update(optimizer, learning_rate, layer_params:dict, gradients:jnp.ndarray, opt_state:dict, *args, **kwargs) -> dict:
    updated_params = {}
    new_opt_state = {}
    
    for name, value in layer_params.items():
      updated_params[name], new_opt_state[name] = optimizer.update(
        learning_rate,
        value,
        gradients[name],
        opt_state[name],
        *args,
        **kwargs
      )
    
    return updated_params, new_opt_state

class Convolution(Layer):
  layer_seed = 0
  training_only=False
  def __init__(self, kernel:tuple[int,int], channels:int, function:Function, stride:tuple[int,int], initializer:Initializer=Default(), name:str="", *args, **kwargs):
    """
    Convolution
    ---------
      a Convolution layer within the context of deep learning is actually cross correlation. 
      Accepts and returns 3D arrays with the shape (Incoming Channels, Image Height, Image Width) excluding the batch dimension.
    ---------
    Args
    ---------
    - kernel                 (tuple[int,int]) : the kernel dimensions to apply, must be of the form (kernel_height, kernel_width)
    - channels               (int)             : the number of channels to output
    - stride                 (tuple[int,int]) : the 2D stride to apply to the kernel
    - function               (Function)        : the activation function
    - (Optional) name        (str)             : the name of the layer
    - (Optional) initializer (Initializer)     : intializer for the weights, defaults to Default
    - (Optional) *args                         : variable length argument list
    - (Optional) **kwargs                      : arbitrary keyword arguments
    """
    self.kernel = kernel
    self.channels = channels
    self.stride = stride
    self.name = name
    self.function = function
    self.initializer = initializer
    self.params = {}
    self.input_shape = None
    self.output_shape = None

  def calibrate(self, fan_in_shape:tuple[int,int,int], fan_out_shape:int) -> tuple[dict, tuple[int,...]]:
    # fan_in_shape = (C_in, H, W)
    C_in, H, W = fan_in_shape

    weights = self.initializer(
      (self.channels, C_in, *self.kernel),
      C_in * self.kernel[0] * self.kernel[1],
      fan_out_shape,
    )
    biases = jnp.zeros((self.channels,))

    # output dims (VALID padding)
    out_H = (H - self.kernel[0]) // self.stride[0] + 1
    out_W = (W - self.kernel[1]) // self.stride[1] + 1

    parametrics = {
      paramater_name: self.initializer(self.layer_seed, (self.channels,), C_in * self.kernel[0] * self.kernel[1], fan_out_shape) for paramater_name in self.function.parameters
    }
    
    return {"weights": weights, "biases": biases, **parametrics}, (self.channels, out_H, out_W)

  def forward(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    inputs: (N, C_in, H, W)
    weights: (C_out, C_in, kH, kW)
    bias: (C_out,)
    """
    if inputs.ndim != 4:
      inputs = jnp.expand_dims(inputs, axis=1)

    convolved = jax.lax.conv_general_dilated(
      lhs=inputs,
      rhs=params["weights"],
      window_strides=self.stride,
      padding="VALID",
      dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )

    bias = params["biases"][jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
    WS = convolved + bias
    activated = self.function.forward(WS, **params)
    return activated, WS

  def backward(self, params:dict, inputs:jnp.ndarray, upstream_error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    if inputs.ndim != 4:
      inputs = jnp.expand_dims(inputs, axis=1)

    all_grads = self.function.backward(upstream_error, weighted_sums, **params)
    
    parametric_gradients = {}
    for name, gradient in all_grads.items():
      if name != 'x':
        parametric_gradients[name] = gradient  # (C_out,)

    d_WS = all_grads['x']  # (N, C_out, H_out, W_out)
    
    # bias gradients
    grad_bias = jnp.sum(d_WS, axis=(0, 2, 3))  # (C_out,)

    def correlate(inputs, errors, kernel_shape, strides):
      N, C_in, H_in, W_in = inputs.shape
      _, C_out, H_out, W_out = errors.shape
      kH, kW = kernel_shape
      sH, sW = strides

      grad_weights = jnp.zeros((C_out, C_in, kH, kW))

      # Loop over the batches
      for n in range(N):
        # Loop over the output spatial dimensions
        for h_out in range(H_out):
          for w_out in range(W_out):
            # Calculate the slice for the input patch
            h_start, w_start = h_out * sH, w_out * sW
            input_patch = jax.lax.slice(
              inputs[n],
              (0, h_start, w_start),
              (C_in, h_start + kH, w_start + kW)
            )

            # Get the error for the current output position
            error_patch = jax.lax.slice(
              errors[n],
              (0, h_out, w_out),
              (C_out, h_out + 1, w_out + 1)
            ).reshape(C_out, 1, 1, 1)

            # Compute the outer product and add to the gradient
            # error_patch: (C_out, 1, 1, 1)
            # input_patch: (C_in, kH, kW)
            # The result has shape (C_out, C_in, kH, kW)
            grad_weights += error_patch * jnp.expand_dims(input_patch, axis=0)

      return grad_weights

    def transposed_convolution(errors, weights, stride=(1,1)):
      N, C_out, H_out, W_out = errors.shape
      C_out_w, C_in, kH, kW = weights.shape
      sH, sW = stride

      # Compute input dimensions
      H_in = (H_out - 1) * sH + kH
      W_in = (W_out - 1) * sW + kW

      upstream_gradient = jnp.zeros((N, C_in, H_in, W_in))

      # Flip weights on spatial dimensions for transposed convolution
      flipped_weights = weights[:, :, ::-1, ::-1]  # shape: (C_out, C_in, kH, kW)

      # Loop over batches, output channels, and spatial positions
      for n in range(N):
        for co in range(C_out):
          for i in range(H_out):
            for j in range(W_out):
              h_start = i * sH
              w_start = j * sW
              # Broadcast the multiplication across input channels
              upstream_gradient = upstream_gradient.at[n, :, h_start:h_start+kH, w_start:w_start+kW].add(
                flipped_weights[co] * errors[n, co, i, j]
              )
      return upstream_gradient

    # Cross-correlation between input and error
    grad_weights = correlate(
      inputs=inputs,
      errors=d_WS,
      kernel_shape=self.kernel,
      strides=self.stride
    )

    # Conv transpose to propagate error back
    upstream_gradient = transposed_convolution(
      errors=d_WS, 
      weights=params["weights"], 
      stride=self.stride
    )

    return upstream_gradient, {"weights": grad_weights, "biases": grad_bias, **parametric_gradients}

  @staticmethod
  def update(optimizer, learning_rate, layer_params:dict, gradients:jnp.ndarray, opt_state:dict, *args, **kwargs) -> dict:
    updated_params = {}
    new_opt_state = {}
    
    for name, value in layer_params.items():
      updated_params[name], new_opt_state[name] = optimizer.update(
        learning_rate,
        value,
        gradients[name],
        opt_state[name],
        *args,
        **kwargs
      )
    
    return updated_params, new_opt_state

class Deconvolution(Layer):
  layer_seed = 0
  training_only=False
  def __init__(self, kernel:tuple[int,int], channels:int, function:Function, stride:tuple[int,int], initializer:Initializer=Default(), name:str="", *args, **kwargs):
    """
    Deconvolution
    ---------
      a Deconvolution layer within the context of deep learning is actually an upscaler, simmilar to Convolution with a flipped direction.
      Accepts and returns 3D arrays with the shape (Incoming Channels, Image Height, Image Width) excluding the batch dimension.
    ---------
    Args
    ---------
    - kernel                 (tuple[int,int]) : the kernel dimensions to apply, must be of the form (kernel_height, kernel_width)
    - channels               (int)             : the number of channels to output
    - stride                 (tuple[int,int]) : the 2D stride to apply to the kernel
    - function               (Function)        : the activation function
    - (Optional) name        (str)             : the name of the layer
    - (Optional) initializer (Initializer)     : intializer for the weights, defaults to Default
    - (Optional) *args                         : variable length argument list
    - (Optional) **kwargs                      : arbitrary keyword arguments
    """
    self.kernel = kernel
    self.channels = channels
    self.name = name
    self.stride = stride
    self.function = function
    self.initializer = initializer
    self.input_shape = None
    self.output_shape = None

  def calibrate(self, fan_in_shape:tuple[int,int, int], fan_out_shape:int) -> tuple[dict, tuple[int,...]]:
    # fan_in_shape = (C_in, H, W)
    C_in, H, W = fan_in_shape
    
    sH, sW = self.stride
    kH, kW = self.kernel
    params = {}
    
    params["weights"] = self.initializer(
      (self.channels, C_in, *self.kernel),
      C_in * kH * kW,
      fan_out_shape,
    )
    out_H = (H + kH) * sH - 1
    out_W = (W + kW) * sW - 1

    params["biases"] = jnp.zeros((self.channels, out_H, out_W))
    
    parametrics = {
      name: self.initializer(self.layer_seed, (self.channels,), C_in * kH * kW, fan_out_shape) for name in self.function.parameters
    }
    return {**params, **parametrics} , (self.channels, out_H, out_W)

  def forward(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    if inputs.ndim != 4:
      inputs = jnp.expand_dims(inputs, axis=1)
      
    N, C_in, H_in, W_in = inputs.shape
    sH, sW = self.stride
    kH, kW = self.kernel
    
    H_out = (H_in - 1) * sH + kH
    W_out = (W_in - 1) * sW + kW
    
    upscaled = jnp.zeros((N, C_in, H_out, W_out))
    
    for n in range(N): # batch
      for co in range(self.channels): # channels out
        for i in range(H_in): # input height
          for j in range(W_in): # input width
            for ci in range(C_in): # channels in
              upscaled = upscaled.at[n,co,i*sH:i*sH+kH,j*sW:j*sW+kW].add(self.params["weights"][co,ci,:,:]*inputs[n,ci,i,j])
              
    WS = upscaled + params["biases"]
    activated = self.function.forward(WS, **params)
    return activated, WS

  def backward(self, params: dict, inputs: jnp.ndarray, upstream_error: jnp.ndarray, weighted_sums: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    if inputs.ndim != 4:
      inputs = jnp.expand_dims(inputs, axis=1)

    all_grads = self.function.backward(upstream_error, weighted_sums, **params)
    
    param_gradients = {}
    for name, gradient in all_grads.items():
      if name != 'x':
        param_gradients[name] = gradient  # (C_out,)
    
    d_WS = all_grads['x']  # (N, C_out, H_out, W_out)

    grad_bias = jnp.sum(d_WS, axis=(0, 2, 3))  # (C_out, H_out, W_out) → reduce to (C_out,)

    def correlate_deconv(inputs, errors, kernel_shape, stride):
      N, C_in, H_in, W_in = inputs.shape
      N_err, C_out, H_out, W_out = errors.shape
      
      kH, kW = kernel_shape
      sH, sW = stride

      grad_weights = jnp.zeros((C_out, C_in, kH, kW))

      for n in range(N):
        for h_in in range(H_in):
          for w_in in range(W_in):
            h_start = h_in * sH
            w_start = w_in * sW
            h_end, w_end = h_start + kH, w_start + kW

            error_patch = errors[n, :, h_start:h_end, w_start:w_end]  # (C_out, kH, kW)
            
            for ci in range(C_in):
              grad_weights = grad_weights.at[:, ci].add(inputs[n, ci, h_in, w_in] * error_patch)
              
      return grad_weights

    def transposed_to_input(errors, weights, stride):
      N, C_out, H_out, W_out = errors.shape
      C_out_w, C_in, kH, kW = weights.shape
      sH, sW = stride

      H_in = (H_out - kH) // sH + 1
      W_in = (W_out - kW) // sW + 1

      upstream_gradient = jnp.zeros((N, C_in, H_in, W_in))

      # No flip for deconv: it's already transposed
      for n in range(N):
        for co in range(C_out):
          for i in range(H_in):
            for j in range(W_in):
              h_start, w_start = i * sH, j * sW
              h_end, w_end = h_start + kH, w_start + kW

              upstream_gradient = upstream_gradient.at[n, :, i, j].add(jnp.sum(weights[co] * errors[n, co, h_start:h_end, w_start:w_end], axis=(1, 2)))
              
      return upstream_gradient

    grad_weights = correlate_deconv(
      inputs=inputs,
      errors=d_WS,
      kernel_shape=self.kernel,
      stride=self.stride
    )

    upstream_gradient = transposed_to_input(
      errors=d_WS,
      weights=params["weights"],
      stride=self.stride
    )

    return upstream_gradient, {"weights": grad_weights, "biases": grad_bias, **param_gradients}

  @staticmethod
  def update(optimizer, learning_rate, layer_params:dict, gradients:jnp.ndarray, opt_state:dict, *args, **kwargs) -> dict:
    updated_params = {}
    new_opt_state = {}
    
    for name, value in layer_params.items():
      updated_params[name], new_opt_state[name] = optimizer.update(
        learning_rate,
        value,
        gradients[name],
        opt_state[name],
        *args,
        **kwargs
      )
    
    return updated_params, new_opt_state

class Recurrent(Layer):
  layer_seed = 0
  training_only=False
  def __init__(self, cells:int, function:Function, input_sequence:tuple[int,...]=None, output_sequence:tuple[int,...]=None, initializer:Initializer=Default(), name:str="", *args, **kwargs):
    """
    Recurrent
    -----
      A Recurrent layer that processes sequences of data, the input format should be a 2D array with an input shape of (sequence_length, features) excluding the batch dimension. 
      The layer maintains the shape of the hidden state throughout the sequence.
    -----
    Args
    -----
    - cells                       (int)           : the number of cells in the layer
    - function                    (Function)      : the activation activation for the layer
    - (Optional) input_sequence   (tuple of int)  : indices of cells that receive input from the input sequence, all cells receive input by default
    - (Optional) output_sequence  (tuple of int)  : indices of cells that output to the next layer, all cells output by default
    - (Optional) initializer      (Initializer)   : intializer for the weights, defaults to Default
    - (Optional) name             (string)        : the name of the layer
    """
    self.cells = cells
    self.name = name
    self.function = function
    self.initializer = initializer
    self.input_sequence = input_sequence
    self.output_sequence = output_sequence

  def calibrate(self, fan_in_shape:tuple[int,...], fan_out_shape:tuple[int,int]) -> tuple[dict, tuple[int,int]]:
    features, sequence_length = fan_in_shape
    
    if self.input_sequence is None:
      self.input_sequence = tuple([_ for _ in range(features)]) 
    if self.output_sequence is None:
      self.output_sequence = tuple([_ for _ in range(self.cells)])

    params = {}
    for cell_index in range(self.cells):
      parametrics ={
        name: self.initializer(self.layer_seed, (features,), features, fan_out_shape[0]) for name in self.function.parameters
      }
      
      params[f'cell_{cell_index}'] = {
        'input_weights': self.initializer(self.layer_seed, (sequence_length,), features, fan_out_shape[0]),
        'carry_weights': self.initializer(self.layer_seed, (sequence_length,), features, fan_out_shape[0]),
        'final_weights': self.initializer(self.layer_seed, (sequence_length * 2, sequence_length), features, fan_out_shape[0]),
        'final_bias': jnp.zeros(sequence_length * 2),
        **parametrics
      }
    return params, (len(self.output_sequence),sequence_length)

  def forward(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # inputs: (batch, seq_len, features)
    batches, seq_len, features = inputs.shape
    per_batch_output = []
    per_batch_WS = []
    
    params = jax.tree.map(
      lambda *arrs: jnp.stack(arrs), # Stacks all arrays from all cells
      *[params[f'cell_{i}'] for i in range(self.cells)]
    )
    
    # arg 1: input carry AND weighted sums
    # arg 2: index AND cell input (input vector)
    @jax.jit
    def cell_calculations(input_carry_and_weighted_sums, index_and_params_and_cell_input):
      input_carry = jnp.where(
        index_and_params_and_cell_input[0] == 0,
        jnp.zeros(features),
        input_carry_and_weighted_sums[0]
      )

      cell_params = jax.tree.map(lambda arr: arr[index_and_params_and_cell_input[0]], params)

      weighted_input = index_and_params_and_cell_input[1] * cell_params['input_weights']
      weighted_carry = input_carry  * cell_params['carry_weights']

      merged = jnp.concatenate((weighted_carry, weighted_input)) + cell_params['final_bias']

      output_carry = self.function.forward(merged @ cell_params['final_weights'], **cell_params)
 
      return [output_carry, merged], output_carry
  
    input_layer_data = [      
      [
        [cell_index, (cell_input if cell_index in self.input_sequence else cell_input)] for cell_index, cell_input in enumerate(batch_input)
      ] for batch_input in inputs
    ]
    
    for batch_input in input_layer_data:
      weighted_sums = []
      outputs = []
      
      output_carry = jnp.zeros(features)
      merged = jnp.zeros(features)
      for cell_input in batch_input:
        (output_carry, merged), output = cell_calculations((output_carry, merged), cell_input)
        
        weighted_sums.append(jnp.array(merged))
        outputs.append(output) if cell_input[0] in self.output_sequence else None
      
      per_batch_output.append(outputs)
      per_batch_WS.append(weighted_sums)
          
    return jnp.array(per_batch_output), per_batch_WS

  # @functools.partial(jax.jit, static_argnums=(0,))
  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    batches, seq_len, features = inputs.shape
    grads = {k: {
      "input_weights": jnp.zeros_like(v["input_weights"]),
      "carry_weights": jnp.zeros_like(v["carry_weights"]),
      "final_weights": jnp.zeros_like(v["final_weights"]),
      "final_bias": jnp.zeros_like(v["final_bias"])
    } for k, v in params.items()}

    input_grads = jnp.zeros_like(inputs)

    for n in range(batches):
      
      grad_carry = jnp.zeros(features)
      for cell_index in reversed(range(self.cells)):
        
        cell_params = params[f'cell_{cell_index}']
        final_input = weighted_sums[n][cell_index]
        
        WS = final_input @ cell_params['final_weights']
        
        local_error = error[n, cell_index] + grad_carry
        
        all_grads = self.function.backward(local_error, WS)
        
        for name, grad in all_grads.items():
          if name != 'x' and name in cell_params:
            grads[f'cell_{cell_index}'][name] = grad
        
        grads_z = all_grads['x']
        
        grads_weights = jnp.outer(final_input, grads_z)  # (in_features, out_features)
        grads_biases = jnp.sum(grads_z, axis=0)  # (out_features,)
        upstream_gradient = grads_z @ cell_params['final_weights'].T  # (batch, in_features)

        if cell_index in self.input_sequence:
          input_feature_idx = self.input_sequence.index(cell_index)
          input_vector = inputs[n, input_feature_idx, :]
        else:
          input_vector = jnp.zeros(seq_len)

        prev_carry = jnp.zeros(features) if cell_index == 0 else weighted_sums[n][cell_index-1]
        
        grads[f'cell_{cell_index}']["input_weights"] += upstream_gradient[:len(grads[f'cell_{cell_index}']["input_weights"])]
        grads[f'cell_{cell_index}']["carry_weights"] += upstream_gradient[len(grads[f'cell_{cell_index}']["input_weights"]):]
        grads[f'cell_{cell_index}']["final_weights"] += grads_weights
        grads[f'cell_{cell_index}']["final_bias"]    += grads_biases

        if cell_index in self.input_sequence:
          input_grads = input_grads.at[n, :, input_feature_idx].add(grads_z @ cell_params['input_weights'].T)

        grad_carry = grads_z @ cell_params['carry_weights'].T

    return input_grads, grads


  @staticmethod
  def update(optimizer, learning_rate, layer_params:dict, gradients:jnp.ndarray, opt_state:dict, *args, **kwargs) -> dict:
    updated_params = {}
    new_opt_state = {}
    
    for cell_key, cell_params in layer_params.items():
      cell_grads = gradients[cell_key]
      new_cell_params = {}
      new_cell_opt_state = {}

      for param_name, param_value in cell_params.items():
        new_cell_params[param_name], new_cell_opt_state[param_name] = optimizer.update(
          learning_rate,
          param_value,
          cell_grads[param_name],
          opt_state[cell_key][param_name],
          *args,
          **kwargs
        )

      updated_params[cell_key] = new_cell_params
      new_opt_state[cell_key] = new_cell_opt_state
    
    return updated_params, new_opt_state

class LSTM(Layer):
  layer_seed = 0
  training_only=False
  def __init__(self, cells:int, function:Function, input_sequence:tuple[int,...]=None, output_sequence:tuple[int,...]=None, initializer:Initializer=Default(), name:str="", *args, **kwargs):
    """
    LSTM (Long Short-Term Memory)
    -----
      A Long Short-Term Memory layer that processes sequences of data, the input format should be a 2D array with an input shape of (sequence_length, features) excluding the batch dimension. 
      The layer maintains the shape of the hidden state throughout the sequence.
    -----
    Args
    -----
    - cells                       (int)           : the number of cells in the layer
    - function                    (Function)      : the activation function for the layer
    - (Optional) input_sequence   (tuple of int)  : indices of cells that receive input from the input sequence, all cells receive input by default
    - (Optional) output_sequence  (tuple of int)  : indices of cells that output to the next layer, all cells output by default
    - (Optional) initializer      (Initializer)   : the initializer for the layer
    - (Optional) name             (string)        : the name of the layer
    """
    self.cells = cells
    self.name = name
    self.function = function
    self.initializer = initializer
    self.input_sequence = input_sequence
    self.output_sequence = output_sequence

  def calibrate(self, fan_in_shape:tuple[int,...], fan_out_shape:tuple[int,int]) -> tuple[dict, tuple[int,...]]:

    sequence_length, features = fan_in_shape

    if sequence_length != self.cells:
      raise ValueError(f"calibrate mismatch: sequence_length={sequence_length} but self.cells={self.cells}")

    if self.input_sequence is None:
      self.input_sequence = tuple([_ for _ in range(self.cells)])
    if self.output_sequence is None:
      self.output_sequence = tuple([_ for _ in range(self.cells)])

    concat_size = features * 2  # merged = concat(weighted_carry, weighted_input)

    params = {}
    for cell_index in range(self.cells):
      parametrics = {
        name: self.initializer(self.layer_seed, (features,), concat_size, features) for name in self.function.parameters
      }
      
      params[f'cell_{cell_index}'] = {
        # elementwise multipliers (preserve length)
        'input_weights': self.initializer(self.layer_seed, (features,), features, features),
        'carry_weights': self.initializer(self.layer_seed, (features,), features, features),

        'forget_weights': self.initializer(self.layer_seed, (concat_size, features), concat_size, features),
        'forget_bias': jnp.zeros((features,)),

        'input_gate_weights': self.initializer(self.layer_seed, (concat_size, features), concat_size, features),
        'input_gate_bias': jnp.zeros((features,)),

        'output_gate_weights': self.initializer(self.layer_seed, (concat_size, features), concat_size, features),
        'output_gate_bias': jnp.zeros((features,)),

        'candidate_weights': self.initializer(self.layer_seed, (concat_size, features), concat_size, features),
        'candidate_bias': jnp.zeros((features,)),

        'final_weights': self.initializer(self.layer_seed, (concat_size, features), concat_size, features),
        'final_bias': jnp.zeros((concat_size,)),
        
        **parametrics
      }
    return params, (len(self.output_sequence),sequence_length)

  def forward(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    batches, cells_dim, features = inputs.shape
    if cells_dim != self.cells:
      raise ValueError(f"apply mismatch: inputs has cells_dim={cells_dim} but self.cells={self.cells}")

    per_batch_output = []
    per_batch_WS = []

    for n in range(batches):
      # initial carry / hidden are zeros (preserve-length)
      input_carry = jnp.zeros((features,))
      c_t = jnp.zeros((features,))    # cell state (kept for completeness, used in gating)
      h_t = jnp.zeros((features,))

      weighted_sums = []
      outputs = []

      for cell_index in range(self.cells):
        cell_params = params[f'cell_{cell_index}']

        # elementwise weighted inputs
        if cell_index in self.input_sequence:
          input_feature_idx = self.input_sequence.index(cell_index)
          input_vector = inputs[n, input_feature_idx, :]
        else:
          input_vector = jnp.zeros((features,))

        weighted_input = input_vector * cell_params['input_weights']   # elementwise
        weighted_carry = input_carry * cell_params['carry_weights']    # elementwise

        # merged concat (carry first, input second) + final_bias (matches your Recurrent idea)
        merged = jnp.concatenate((weighted_carry, weighted_input)) + cell_params['final_bias']  # (2*features,)

        # gates: merged -> features (fully connected preserving length)
        z_f = jnp.dot(merged, cell_params['forget_weights']) + cell_params['forget_bias']
        z_i = jnp.dot(merged, cell_params['input_gate_weights']) + cell_params['input_gate_bias']
        z_g = jnp.dot(merged, cell_params['candidate_weights']) + cell_params['candidate_bias']
        z_o = jnp.dot(merged, cell_params['output_gate_weights']) + cell_params['output_gate_bias']

        f_t = jax.nn.sigmoid(z_f)
        i_t = jax.nn.sigmoid(z_i)
        o_t = jax.nn.sigmoid(z_o)
        g_t = jax.nn.tanh(z_g)

        # cell + hidden update (standard LSTM equations but using merged->gates)
        c_prev = c_t
        h_prev = h_t

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * jax.nn.tanh(c_t)

        # final FC head (merged -> features) then FUNCTION (pointwise)
        act_WS = jnp.dot(merged, cell_params['final_weights'])   # (features,)
        output_carry = self.function.forward(act_WS, **cell_params)

        # stash all needed values for backward
        WS = {
          'merged': merged, 
          'act_WS': act_WS,
          
          'z_f': z_f,                                                                                                                                               
          'z_i': z_i, 
          'z_o': z_o, 
          'z_g': z_g,
          
          'f_t': f_t, 
          'i_t': i_t, 
          'o_t': o_t, 
          'g_t': g_t,
          
          'c_prev': c_prev, 
          'c_t': c_t,
          
          'h_prev': h_prev, 
          'h_t': h_t,
          
          'input_vector': input_vector,
          'weighted_input': weighted_input, 'weighted_carry': weighted_carry
        }
        weighted_sums.append(WS)

        # collect outputs in order of output_sequence (same as your Recurrent)
        if cell_index in self.output_sequence:
          outputs.append(output_carry)

        # next cell's carry / hidden
        input_carry = output_carry   # keep the same naming as your Recurrent

      per_batch_output.append(jnp.stack(outputs, axis=0) if len(outputs) > 0 else jnp.zeros((0, features)))
      per_batch_WS.append(weighted_sums)

    return jnp.stack(per_batch_output, axis=0), per_batch_WS

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:list) -> tuple[jnp.ndarray, dict]:
    batches, cells_dim, features = inputs.shape
    if cells_dim != self.cells:
      raise ValueError(f"backward mismatch: inputs has cells_dim={cells_dim} but self.cells={self.cells}")

    # prepare grads per-cell
    grads = {k: {
      "input_weights": jnp.zeros_like(v["input_weights"]),
      "carry_weights": jnp.zeros_like(v["carry_weights"]),
      
      "forget_weights": jnp.zeros_like(v["forget_weights"]),
      "forget_bias": jnp.zeros_like(v["forget_bias"]),
      
      "input_gate_weights": jnp.zeros_like(v["input_gate_weights"]),
      "input_gate_bias": jnp.zeros_like(v["input_gate_bias"]),
      
      "output_gate_weights": jnp.zeros_like(v["output_gate_weights"]),
      "output_gate_bias": jnp.zeros_like(v["output_gate_bias"]),
      
      "candidate_weights": jnp.zeros_like(v["candidate_weights"]),
      "candidate_bias": jnp.zeros_like(v["candidate_bias"]),
      
      "final_weights": jnp.zeros_like(v["final_weights"]),
      "final_bias": jnp.zeros_like(v["final_bias"]),
    } for k, v in params.items()}

    input_grads = jnp.zeros_like(inputs)

    for n in range(batches):
      # gradient carrying along the chain (for hidden / carry)
      dh_next = jnp.zeros((features,))   # gradient w.r.t. next hidden (from future cell)
      dc_next = jnp.zeros((features,))   # gradient w.r.t. next cell state

      # iterate cells backward
      for cell_index in reversed(range(self.cells)):
        cell_key = f'cell_{cell_index}'
        cell_params = params[cell_key]
        WS = weighted_sums[n][cell_index]

        merged = WS['merged']        # 2 * features
        act_WS = WS['act_WS']        # features
        
        z_f = WS['z_f']; z_i = WS['z_i']; z_o = WS['z_o']; z_g = WS['z_g']
        f_t = WS['f_t']; i_t = WS['i_t']; o_t = WS['o_t']; g_t = WS['g_t']
        
        c_prev = WS['c_prev']; c_t = WS['c_t']
        h_prev = WS['h_prev']; h_t = WS['h_t']
        
        input_vector = WS['input_vector']

        # get error slice if this cell produced an output
        if cell_index in self.output_sequence:
          k = self.output_sequence.index(cell_index)
          d_y = error[n, k, :]
        else:
          d_y = jnp.zeros((features,))

        # grads_z corresponds to delta in feature space after FUNCTION
        all_grads = self.function.backward(d_y, act_WS, **cell_params)   # (features,)
        
        for name, gradient in all_grads.items():
          if name != 'x':
            grads[cell_key][name] = gradient
        
        grads_z = all_grads['x']
        
        # final_weights gradient
        grads[cell_key]['final_weights'] += jnp.outer(merged, grads_z)   # (2*features, features)

        # gradient flowing back into merged from final head
        upstream_from_final = jnp.dot(grads_z, cell_params['final_weights'].T)  # (2*features,)

        grads[cell_key]['final_bias'] += upstream_from_final

        # compute derivative wrt o, i, f, g using standard LSTM formulas
        tanh_c = jax.nn.tanh(c_t)
        # output gate
        do = dh_next * tanh_c
        do_raw = do * (o_t * (1.0 - o_t))   # σ'(z_o) with o_t

        # cell state derivative
        dc = dh_next * o_t * (1.0 - tanh_c ** 2) + dc_next

        di_raw = (dc * g_t) * (i_t * (1.0 - i_t))   # σ'(z_i)
        df_raw = (dc * c_prev) * (f_t * (1.0 - f_t))   # σ'(z_f)
        dg_raw = (dc * i_t) * (1.0 - g_t ** 2)       # tanh'(z_g) via g_t

        # accumulate gradients for gate FCs (merged -> features)
        grads[cell_key]['forget_weights'] += jnp.outer(merged, df_raw)
        grads[cell_key]['forget_bias'] += df_raw

        grads[cell_key]['input_gate_weights'] += jnp.outer(merged, di_raw)
        grads[cell_key]['input_gate_bias'] += di_raw

        grads[cell_key]['output_gate_weights'] += jnp.outer(merged, do_raw)
        grads[cell_key]['output_gate_bias'] += do_raw

        grads[cell_key]['candidate_weights'] += jnp.outer(merged, dg_raw)
        grads[cell_key]['candidate_bias'] += dg_raw

        # gradient wrt merged coming from gate paths (via their weight matrices)
        d_concat_from_gates = (
          jnp.dot(df_raw, cell_params['forget_weights'].T) +
          jnp.dot(di_raw, cell_params['input_gate_weights'].T) +
          jnp.dot(do_raw, cell_params['output_gate_weights'].T) +
          jnp.dot(dg_raw, cell_params['candidate_weights'].T)
        )   # shape (2*features,)

        # total gradient into merged = from final head + from gates
        d_merged_total = upstream_from_final + d_concat_from_gates   # (2*features,)

        # split into carry and input halves
        d_weighted_carry = d_merged_total[:features]
        d_weighted_input = d_merged_total[features:]

        # weighted_input = input_vector * input_weights  -> d input_weights = input_vector * d_weighted_input
        grads[cell_key]['input_weights'] += input_vector * d_weighted_input
        grads[cell_key]['carry_weights'] += h_prev * d_weighted_carry

        if cell_index in self.input_sequence:
          input_feature_idx = self.input_sequence.index(cell_index)
          # derivative: weighted_input = input_vector * input_weights -> dx = d_weighted_input * input_weights
          input_grads = input_grads.at[n, input_feature_idx, :].add(d_weighted_input * cell_params['input_weights'])

        dh_prev_from_merged = d_weighted_carry * cell_params['carry_weights']

        dh_next = dh_prev_from_merged
        dc_next = dc * f_t

    return input_grads, grads

  @staticmethod
  def update(optimizer, learning_rate, layer_params:dict, gradients:jnp.ndarray, opt_state:dict, *args, **kwargs) -> dict:
    updated_params = {}
    new_opt_state = {}
    
    for cell_key, cell_params in layer_params.items():
      cell_grads = gradients[cell_key]
      new_cell_params = {}
      new_cell_opt_state = {}

      for param_name, param_value in cell_params.items():
        new_cell_params[param_name], new_cell_opt_state[param_name] = optimizer.update(
          learning_rate,
          param_value,
          cell_grads[param_name],
          opt_state[cell_key][param_name],
          *args,
          **kwargs
        )

      updated_params[cell_key] = new_cell_params
      new_opt_state[cell_key] = new_cell_opt_state
    
    return updated_params, new_opt_state

class GRU(Layer):
  layer_seed = 0
  training_only=False
  def __init__(self, cells:int, function:Function, input_sequence:tuple[int,...]=None, output_sequence:tuple[int,...]=None, initializer:Initializer=Default(), name:str="", *args, **kwargs):
    """
    GRU (Gated Recurrent Unit)
    -----
      A Gated Recurrent Unit layer that processes sequences of data, the input format should be a 2D array with an input shape of (sequence_length, features) excluding the batch dimension. 
      The layer maintains the shape of the hidden state throughout the sequence.
    -----
    Args
    -----
    - cells                       (int)           : the number of cells in the layer
    - function                    (Function)      : the activation function for the layer
    - (Optional) input_sequence   (tuple of int)  : indices of cells that receive input from the input sequence, all cells receive input by default
    - (Optional) output_sequence  (tuple of int)  : indices of cells that output to the next layer, all cells output by default
    - (Optional) initializer      (Initializer)   : the initializer for the layer
    - (Optional) name             (string)        : the name of the layer
    """
    self.cells = cells
    self.name = name
    self.function = function
    self.initializer = initializer
    self.input_sequence = input_sequence
    self.output_sequence = output_sequence

  def calibrate(self, fan_in_shape:tuple[int,...], fan_out_shape:tuple[int,int]) -> tuple[dict, tuple[int,...]]:
    sequence_length, features = fan_in_shape

    if self.input_sequence is None:
      self.input_sequence = tuple([_ for _ in range(features)])
    if self.output_sequence is None:
      self.output_sequence = tuple([_ for _ in range(self.cells)])

    params = {}
    for cell_index in range(self.cells):
      parametrics = {}
      for name in self.function.parameters:
        parametrics[name] = self.initializer(self.layer_seed, (features,), features, features)

      params[f'cell_{cell_index}'] = {
        # reset gate weights: x @ W_r  and h_prev @ U_r
        "W_r": self.initializer(self.layer_seed, (features, features), features, features),
        "U_r": self.initializer(self.layer_seed, (features, features), features, features),
        "b_r": jnp.zeros((features,)),

        # update gate
        "W_z": self.initializer(self.layer_seed, (features, features), features, features),
        "U_z": self.initializer(self.layer_seed, (features, features), features, features),
        "b_z": jnp.zeros((features,)),

        # candidate hidden
        "W_h": self.initializer(self.layer_seed, (features, features), features, features),
        "U_h": self.initializer(self.layer_seed, (features, features), features, features),
        "b_h": jnp.zeros((features,)),

        # final fully-connected layer before output (concat(h, x) -> features)
        "final_weights": self.initializer(self.layer_seed, (features * 2, features), features, features),
        "final_bias": jnp.zeros((features,)),

        **parametrics
      }

    # return params and output shape (len(output_sequence) used elsewhere), but keep consistent with prior API
    return params, (len(self.output_sequence), features)

  def forward(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    batches, seq_len, features = inputs.shape
    per_batch_output = []
    per_batch_WS = []

    for n in range(batches):
      h_t = jnp.zeros((features,))   # hidden state (preserve-length)
      weighted_sums = []
      outputs = []

      for cell_index in range(self.cells):
        cell_params = params[f'cell_{cell_index}']

        # input for this cell (or zeros if not in input_sequence)
        if cell_index in self.input_sequence:
          input_feature_idx = self.input_sequence.index(cell_index)
          x_t = inputs[n, input_feature_idx, :]
        else:
          x_t = jnp.zeros((features,))

        # Reset gate: r_t = sigmoid(xW_r + h_prev U_r + b_r)
        pre_r = jnp.dot(x_t, cell_params["W_r"]) + jnp.dot(h_t, cell_params["U_r"]) + cell_params["b_r"]
        r_t = jax.nn.sigmoid(pre_r)

        # Update gate: z_t = sigmoid(xW_z + h_prev U_z + b_z)
        pre_z = jnp.dot(x_t, cell_params["W_z"]) + jnp.dot(h_t, cell_params["U_z"]) + cell_params["b_z"]
        z_t = jax.nn.sigmoid(pre_z)

        # Candidate hidden: h_hat = FUNCTION(xW_h + (r_t * h_prev) U_h + b_h)
        rh = r_t * h_t                            # elementwise
        pre_h_hat = jnp.dot(x_t, cell_params["W_h"]) + jnp.dot(rh, cell_params["U_h"]) + cell_params["b_h"]
        # support both callable FUNCTION function and FUNCTION object
        h_hat = self.function.forward(pre_h_hat, **cell_params)

        # New hidden state
        h_new = (1.0 - z_t) * h_t + z_t * h_hat

        # Final head: concat(h_new, x_t) -> final_weights -> FUNCTION -> output
        merged = jnp.concatenate((h_new, x_t))   # (2*features,)
        act_preact = jnp.dot(merged, cell_params["final_weights"]) + cell_params["final_bias"]
        output_carry = self.function.forward(act_preact, **cell_params)

        # Save everything needed for backward
        WS = {
          "x_t": x_t,
          "h_prev": h_t,
          "pre_r": pre_r, "r_t": r_t,
          "pre_z": pre_z, "z_t": z_t,
          "pre_h_hat": pre_h_hat, "h_hat": h_hat,
          "h_new": h_new,
          "merged": merged, "act_preact": act_preact
        }
        weighted_sums.append(WS)

        # If this cell is an output cell, collect output
        if cell_index in self.output_sequence:
          outputs.append(output_carry)

        # advance hidden
        h_t = h_new

      per_batch_output.append(jnp.stack(outputs, axis=0) if len(outputs) > 0 else jnp.zeros((0, features)))
      per_batch_WS.append(weighted_sums)

    return jnp.stack(per_batch_output, axis=0), per_batch_WS

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    batches, seq_len, features = inputs.shape

    # init grads per cell
    grads = {k: {
      "W_r": jnp.zeros_like(v["W_r"]),
      "U_r": jnp.zeros_like(v["U_r"]),
      "b_r": jnp.zeros_like(v["b_r"]),

      "W_z": jnp.zeros_like(v["W_z"]),
      "U_z": jnp.zeros_like(v["U_z"]),
      "b_z": jnp.zeros_like(v["b_z"]),

      "W_h": jnp.zeros_like(v["W_h"]),
      "U_h": jnp.zeros_like(v["U_h"]),
      "b_h": jnp.zeros_like(v["b_h"]),

      "final_weights": jnp.zeros_like(v["final_weights"]),
      "final_bias": jnp.zeros_like(v["final_bias"])
    } for k, v in params.items()}

    input_grads = jnp.zeros_like(inputs)

    # process each batch independently
    for n in range(batches):
      dh_next = jnp.zeros((features,))   # gradient w.r.t. next hidden (in chain)
      dc_unused = None                   # GRU has no cell state, placeholder

      # iterate cells backward
      for cell_index in reversed(range(self.cells)):
        cell_key = f'cell_{cell_index}'
        cell_params = params[cell_key]
        WS = weighted_sums[n][cell_index]

        # get forward cached values
        x_t = WS["x_t"]
        h_prev = WS["h_prev"]
        pre_r, r_t = WS["pre_r"], WS["r_t"]
        pre_z, z_t = WS["pre_z"], WS["z_t"]
        pre_h_hat, h_hat = WS["pre_h_hat"], WS["h_hat"]
        h_new = WS["h_new"]
        merged = WS["merged"]
        act_preact = WS["act_preact"]

        # error slice for this cell (if it's an output)
        if cell_index in self.output_sequence:
          k = self.output_sequence.index(cell_index)
          d_out = error[n, k, :]
        else:
          d_out = jnp.zeros((features,))

        # --- Backprop through final FC head ---
        # derivative through FUNCTION at final head
        d_act = self.function.backward(d_out, act_preact, **cell_params)['x']
        grads[cell_key]["final_weights"] += jnp.outer(merged, d_act)
        grads[cell_key]["final_bias"] += d_act
        # upstream into merged (2*features,)
        d_merged = jnp.dot(d_act, cell_params["final_weights"].T)  # (2*features,)

        # split merged grads into dh_new and dx contributions
        d_h_from_final = d_merged[:features]
        d_x_from_final = d_merged[features:]

        # total gradient w.r.t. h_new includes dh_next from later cells (temporal) and final head
        dh_total = d_h_from_final + dh_next

        # --- GRU recurrence backprop ---
        # h_new = (1 - z_t) * h_prev + z_t * h_hat
        # contributions:
        dh_hat = dh_total * z_t                   # (features,)
        dz = dh_total * (h_hat - h_prev)          # (features,)
        dh_prev_part = dh_total * (1.0 - z_t)     # part that flows to previous hidden directly

        # backprop through h_hat = FUNCTION(pre_h_hat)
        all_grads = self.function.backward(dh_hat, pre_h_hat, **cell_params)
        
        for name, gradient in all_grads.items():
          if name != 'x':   # 'x' is the gradient wrt pre_h_hat
            if name in cell_params:
              grads[cell_key][name] = gradient
        
        d_pre_h_hat = all_grads['x']
        
        # grads to W_h, U_h, b_h
        grads[cell_key]["W_h"] += jnp.outer(x_t, d_pre_h_hat)
        grads[cell_key]["U_h"] += jnp.outer(r_t * h_prev, d_pre_h_hat)
        grads[cell_key]["b_h"] += d_pre_h_hat

        # propagate to x and to (r * h_prev)
        d_x = jnp.dot(d_pre_h_hat, cell_params["W_h"].T)   # from W_h path
        d_rh = jnp.dot(d_pre_h_hat, cell_params["U_h"].T)  # gradient into elementwise (r * h_prev)

        # d_rh contributes to r and h_prev:
        d_r_from_hpath = d_rh * h_prev
        d_hprev_from_hpath = d_rh * r_t

        # --- reset gate r backprop: r = sigmoid(pre_r)
        # pre_r receives contributions from both d_r_from_hpath and possibly elsewhere (here only from h_hat path)
        d_pre_r = d_r_from_hpath * (r_t * (1.0 - r_t))   # sigmoid'
        grads[cell_key]["W_r"] += jnp.outer(x_t, d_pre_r)
        grads[cell_key]["U_r"] += jnp.outer(h_prev, d_pre_r)
        grads[cell_key]["b_r"] += d_pre_r

        # propagate to x and h_prev from reset gate path
        d_x += jnp.dot(d_pre_r, cell_params["W_r"].T)
        d_hprev_from_rpath = jnp.dot(d_pre_r, cell_params["U_r"].T)

        # --- update gate z backprop: z = sigmoid(pre_z)
        d_pre_z = dz * (z_t * (1.0 - z_t))
        grads[cell_key]["W_z"] += jnp.outer(x_t, d_pre_z)
        grads[cell_key]["U_z"] += jnp.outer(h_prev, d_pre_z)
        grads[cell_key]["b_z"] += d_pre_z

        d_x += jnp.dot(d_pre_z, cell_params["W_z"].T)
        d_hprev_from_zpath = jnp.dot(d_pre_z, cell_params["U_z"].T)

        # --- accumulate gradient wrt h_prev for passing to previous timestep
        dh_prev = dh_prev_part + d_hprev_from_hpath + d_hprev_from_rpath + d_hprev_from_zpath

        # --- input gradient: include contribution from final head dx_from_final and from gate/candidate paths (d_x)
        total_dx = d_x + d_x_from_final

        # write input gradient if this cell consumes input
        if cell_index in self.input_sequence:
          input_idx = self.input_sequence.index(cell_index)
          input_grads = input_grads.at[n, input_idx, :].add(total_dx)

        # set dh_next for the previous cell (next iteration of reversed loop)
        dh_next = dh_prev

    return input_grads, grads

  @staticmethod
  def update(optimizer, learning_rate, layer_params:dict, gradients:jnp.ndarray, opt_state:dict, *args, **kwargs) -> dict:
    updated_params = {}
    new_opt_state = {}
    
    for cell_key, cell_params in layer_params.items():
      cell_grads = gradients[cell_key]
      new_cell_params = {}
      new_cell_opt_state = {}

      for param_name, param_value in cell_params.items():
        new_cell_params[param_name], new_cell_opt_state[param_name] = optimizer.update(
          learning_rate,
          param_value,
          cell_grads[param_name],
          opt_state[cell_key][param_name],
          *args,
          **kwargs
        )

      updated_params[cell_key] = new_cell_params
      new_opt_state[cell_key] = new_cell_opt_state
    
    return updated_params, new_opt_state

class Attention(Layer):
  layer_seed = 0
  training_only=False
  def __init__(self, heads:int, function:Function, initializer:Initializer=Default(), name:str="", *args, **kwargs):
    """
    Multiheaded Self-Attention
    -----
      Primary block within Transformer networks, the amount of attention heads is configurable. 
      It accepts data with shape (batch_size, sequence_length, features) simmilar to RNNs.
    -----
    Args
    -----
    - heads                  (int)          : the number of attention heads
    - function               (Function)     : the activation function for the layer
    - (Optional) initializer (Initializer)  : the initializer for the layer
    - (Optional) name        (string)       : the name of the layer
    """
    self.heads = heads
    self.name = name
    self.function = function
    self.initializer = initializer

  def calibrate(self, fan_in_shape:tuple[int,...], fan_out_shape:tuple[int,int]) -> tuple[dict, tuple[int,int]]:
    """
    fan_in_shape: (features, sequence_length)
    fan_out_shape: (features_out, sequence_length_out)
    """
    features, sequence_length = fan_in_shape
    params = {}

    # each head has its own Q, K, V
    for head in range(self.heads):
      params[f'head_{head}'] = {
        "W_Q": self.initializer(self.layer_seed, (sequence_length, sequence_length), features, fan_out_shape[0]),
        "W_K": self.initializer(self.layer_seed, (sequence_length, sequence_length), features, fan_out_shape[0]),
        "W_V": self.initializer(self.layer_seed, (sequence_length, sequence_length), features, fan_out_shape[0]),
        "b_Q": jnp.zeros(sequence_length),
        "b_K": jnp.zeros(sequence_length),
        "b_V": jnp.zeros(sequence_length),
      }

    # final projection after concat
    params["final"] = {
      "W_O": self.initializer(self.layer_seed, (sequence_length * self.heads, sequence_length), features, fan_out_shape[0]),
      "b_O": jnp.zeros(sequence_length)
    }
    
    parametrics = {
      parameter_name: self.initializer(self.layer_seed, (sequence_length,), features, fan_out_shape[0]) for parameter_name in self.function.parameters
    }
    
    return {**params, **parametrics}, (features, sequence_length)

  def forward(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch_size, features, seq_len = inputs.shape
    per_batch_outputs = []
    per_batch_WS = []

    # for features, sequence in zip(inputs)
    for n in range(batch_size):  # process one batch at a time
      batch_input = inputs[n]  # (seq_len, features)
      head_outputs = []
      head_ws = []

      for head in range(self.heads):
        head_params = params[f'head_{head}']

        # fully-connected projections
        Q = batch_input @ head_params["W_Q"] + head_params["b_Q"]  # (S, F)
        K = batch_input @ head_params["W_K"] + head_params["b_K"]  # (S, F)
        V = batch_input @ head_params["W_V"] + head_params["b_V"]  # (S, F)

        # scaled dot-product attention
        attn_scores = (Q @ K.T) / jnp.sqrt(seq_len)     # (S, S)
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)  # (S, S)
        head_out = attn_weights @ V                      # (S, F)

        head_outputs.append(head_out)
        head_ws.append((Q, K, V, attn_scores, attn_weights, head_out))

      # concat all heads along feature dim → (S, F * heads)
      concat_out = jnp.concatenate(head_outputs, axis=-1)
      
      # final FC projection (maps back to (S, F))
      merged = jax.vmap(lambda token: token @ params["final"]["W_O"] + params["final"]["b_O"])(concat_out)
      out = self.function.forward(merged, **params)  # (S, F)

      per_batch_outputs.append(out)
      per_batch_WS.append((head_ws, concat_out, merged, out))

    return jnp.array(per_batch_outputs), per_batch_WS

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:dict) -> tuple[jnp.ndarray, dict]:
    batch_size, features, seq_len = inputs.shape

    # initialize grads
    grads = {}
    for head in range(self.heads):
      grads[f'head_{head}'] = {
        "W_Q": jnp.zeros_like(params[f'head_{head}']["W_Q"]),
        "b_Q": jnp.zeros_like(params[f'head_{head}']["b_Q"]),
        "W_K": jnp.zeros_like(params[f'head_{head}']["W_K"]),
        "b_K": jnp.zeros_like(params[f'head_{head}']["b_K"]),
        "W_V": jnp.zeros_like(params[f'head_{head}']["W_V"]),
        "b_V": jnp.zeros_like(params[f'head_{head}']["b_V"]),
      }

    grads["final"] = {
      "W_O": jnp.zeros_like(params["final"]["W_O"]),
      "b_O": jnp.zeros_like(params["final"]["b_O"]),
    }

    input_grads = jnp.zeros_like(inputs)
    upstream_grads = jnp.zeros_like(inputs)

    # process per batch
    for n in range(batch_size):
      batch_input = inputs[n]    # (S, F)
      batch_error = error[n]     # (S, F)
      head_ws, concat_out, merged, out = weighted_sums[n]

      # ---- backprop through final FC + FUNCTION ----
      all_grads = self.function.backward(batch_error, merged, **params)
      
      parametrics = {}
      for name, grad in all_grads.items():
        if name != "x":
          parametrics[name] = grad
      
      d_out = all_grads['x']   # (S, F)
      
      grads["final"]["W_O"] += jnp.einsum("bi,bj->ij", concat_out, d_out)                # (F*H, F)
      grads["final"]["b_O"] += jnp.sum(d_out, axis=0)              # (F,)
      d_concat = d_out @ params["final"]["W_O"].T                  # (S, F*H)

      # ---- split gradient across heads ----
      split_d = jnp.split(d_concat, self.heads, axis=-1)

      # accumulate grads per-head + input grads
      d_input_total = jnp.zeros_like(batch_input)

      for head, d_head_out in enumerate(split_d):
        Q, K, V, attn_scores, attn_weights, head_out = head_ws[head]
        head_params = params[f'head_{head}']

        # grad wrt V (from attn_weights @ V)
        dV = attn_weights.T @ d_head_out  # (S, F)

        # grad wrt attn_weights
        d_attn_weights = d_head_out @ V.T  # (S, S)

        # grad wrt attn_scores (softmax backprop)
        # softmax jacobian simplified: dL/dS = (grad - sum(grad*W, axis=-1)) * W
        d_attn_scores = d_attn_weights * attn_weights - attn_weights * jnp.sum(d_attn_weights * attn_weights, axis=-1, keepdims=True)

        # grad wrt Q and K
        dQ = d_attn_scores @ K / jnp.sqrt(seq_len)
        dK = d_attn_scores.T @ Q / jnp.sqrt(seq_len)

        # propagate through FCs: (batch_input @ W + b)
        grads[f'head_{head}']["W_Q"] += batch_input.T @ dQ
        grads[f'head_{head}']["b_Q"] += jnp.sum(dQ, axis=0)

        grads[f'head_{head}']["W_K"] += batch_input.T @ dK
        grads[f'head_{head}']["b_K"] += jnp.sum(dK, axis=0)

        grads[f'head_{head}']["W_V"] += batch_input.T @ dV
        grads[f'head_{head}']["b_V"] += jnp.sum(dV, axis=0)

        # accumulate gradient wrt inputs
        d_input_total += dQ @ head_params["W_Q"].T
        d_input_total += dK @ head_params["W_K"].T
        d_input_total += dV @ head_params["W_V"].T

      # assign input grads for this batch
      upstream_grads = upstream_grads.at[n].set(d_input_total)

    return upstream_grads, {**grads, **parametrics}

  @staticmethod
  def update(optimizer, learning_rate, layer_params:dict, gradients:jnp.ndarray, opt_state:dict, *args, **kwargs) -> dict:
    updated_params = {}
    new_opt_state = {}
    
    for cell_key, cell_params in layer_params.items():
      
      cell_grads = gradients[cell_key]
      new_cell_params = {}
      new_cell_opt_state = {}

      for param_name, param_value in cell_params.items():
        new_cell_params[param_name], new_cell_opt_state[param_name] = optimizer.update(
          learning_rate,
          param_value,
          cell_grads[param_name],
          opt_state[cell_key][param_name],
          *args,
          **kwargs
        )

      updated_params[cell_key] = new_cell_params
      new_opt_state[cell_key] = new_cell_opt_state
    
    return updated_params, new_opt_state

# functional layers

class MaxPooling(Layer):
  layer_seed = 0
  training_only=False
  def __init__(self, pool_size:tuple[int,int], strides:tuple[int,int], name:str="", *args, **kwargs):
    """
    Max Pooling
    -----
      A layer that performs max pooling on a 2D input while adjusting to channel dimensions. 
      accepts a 3D/2D input shape of (Channels, Height, Width).
    -----
    Args
    -----
    - pool_size       (tuple of int)  : the size of the pooling window (height, width)
    - strides         (tuple of int)  : the stride of the pooling window (height, width)
    - (Optional) name (string)        : the name of the layer
    """
    self.pool_size = pool_size
    self.strides = strides
    self.name = name

  def calibrate(self, fan_in_shape:tuple[int,...], fan_out_shape:tuple[int,int]) -> tuple[dict, tuple[int,...]]:
    C, H, W = fan_in_shape
    pooled_H = (H - self.pool_size[0]) // self.strides[0] + 1
    pooled_W = (W - self.pool_size[1]) // self.strides[1] + 1
    self.input_shape = fan_in_shape
    
    return {}, (C, pooled_H, pooled_W)

  def forward(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    if len(inputs.shape) != 4:
      inputs = jnp.expand_dims(inputs, axis=1)

    pooled_output = jax.lax.reduce_window(
      inputs,
      init_value=-jnp.inf,
      computation=jax.lax.max,
      window_dimensions=(1, 1, *self.pool_size),
      window_strides=(1, 1, *self.strides),
      padding='VALID'
    )
    
    return pooled_output, inputs

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    if len(inputs.shape) != 4:
      inputs = jnp.expand_dims(inputs, axis=1)
      
    _, _, pooled_H, pooled_W = error.shape

    pool_H, pool_W = self.pool_size
    stride_H, stride_W = self.strides

    def grad_single(x, grad_out):
      """Backprop a single (H,W) map with pooled grads."""
      grad_in = jnp.zeros_like(x)
      for i in range(pooled_H):
        for j in range(pooled_W):
          h_start, h_end = i * stride_H, i * stride_H + pool_H
          w_start, w_end = j * stride_W, j * stride_W + pool_W
          window = x[h_start:h_end, w_start:w_end]
          mask = window == jnp.max(window)
          grad_in = grad_in.at[h_start:h_end, w_start:w_end].add(mask * grad_out[i, j])
      return grad_in
    
    grad = jax.vmap(jax.vmap(grad_single, in_axes=(0,0)), in_axes=(0,0))
    upstream_gradient = grad(inputs, error)
    return upstream_gradient, {}

class MeanPooling(Layer):
  layer_seed = 0
  training_only=False
  def __init__(self, pool_size:tuple[int,int]=(2,2), strides:tuple[int,int]=(2,2), name:str="", *args, **kwargs):
    """
    Mean Pooling
    -----
      A layer that performs mean pooling on a 2D input while adjusting to channel dimensions. 
      accepts a 3D/2D input shape of (Channels, Height, Width).
    -----
    Args
    -----
    - pool_size       (tuple of int)  : the size of the pooling window (height, width)
    - strides         (tuple of int)  : the stride of the pooling window (height, width)
    - (Optional) name (string)        : the name of the layer
    """
    self.pool_size = pool_size
    self.strides = strides
    self.name = name

  def calibrate(self, fan_in_shape:tuple[int,...], fan_out_shape:tuple[int,int]) -> tuple[dict, tuple[int,...]]:
    C, H, W = fan_in_shape
    pooled_H = (H - self.pool_size[0]) // self.strides[0] + 1
    pooled_W = (W - self.pool_size[1]) // self.strides[1] + 1
    self.input_shape = fan_in_shape
    return {}, (C, pooled_H, pooled_W)

  def forward(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    if len(inputs.shape) != 4:
      inputs = jnp.expand_dims(inputs, axis=1)

    pooled_output = jax.lax.reduce_window(
      inputs,
      init_value=0.,
      computation=jax.lax.add,
      window_dimensions=(1, 1, *self.pool_size),
      window_strides=(1, 1, *self.strides),
      padding='VALID'
    )
    pool_area = self.pool_size[0] * self.pool_size[1]
    pooled_output /= pool_area
    return pooled_output, inputs

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    N, C, H, W = inputs.shape
    _, _, pooled_H, pooled_W = error.shape
    pool_H, pool_W = self.pool_size
    stride_H, stride_W = self.strides
    pool_area = pool_H * pool_W

    def grad_single(grad_out):
      grad_in = jnp.zeros((H, W))
      for i in range(pooled_H):
        for j in range(pooled_W):
          h_start, h_end = i * stride_H, i * stride_H + pool_H
          w_start, w_end = j * stride_W, j * stride_W + pool_W
          grad_in = grad_in.at[h_start:h_end, w_start:w_end].add(grad_out[i, j] / pool_area)
      return grad_in

    grad = jax.vmap(jax.vmap(grad_single, in_axes=0), in_axes=0)
    upstream_gradient = grad(error)
    return upstream_gradient, {}

class Flatten(Layer):
  layer_seed = 0
  training_only=False
  def __init__(self, name:str="", *args, **kwargs):
    """
    Flatten
    -----
      A layer that flattens any ndim input into a 1D while accounting for the batch dimension. 
    -----
    Args
    -----
    - (Optional) name (string) : the name of the layer
    """
    
    self.name = name

  def calibrate(self, fan_in_shape:tuple[int,...], fan_out_shape:int) -> tuple[dict, int]:
    flattened_size = jnp.prod(jnp.array(fan_in_shape))
    self.input_shape = fan_in_shape
    
    return {}, (int(flattened_size),)

  def forward(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # Flatten batch to (batch, features)
    flattened_output = inputs.reshape(inputs.shape[0], -1)
    
    return flattened_output, inputs  # Keep inputs for backprop

  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    upstream_gradient = error.reshape(inputs.shape)
    return upstream_gradient, {}

class Operation(Layer):
  layer_seed = 0
  training_only=False
  def __init__(self, function:Function, name:str="", *args, **kwargs):
    """
    Operation
    -----
      A layer that performs an operation on any ndim input while preserving shape. 
      this layer automatically adjusts and does not need to a fixed input shape, but make sure to set the input shape in the format that the operation expects.
    """
    self.function = function
    self.name = name
  
  def calibrate(self, fan_in_shape:tuple[int,...], fan_out_shape:tuple[int,...]) -> tuple[dict, tuple[int,...]]:
    
    # idk if there should be basic checking in the standard layer or if that is the API's spesific responsibility
    
    if self.operation(jnp.zeros(fan_in_shape)).shape != fan_in_shape:
      raise SystemError(f"Operation layer '{self.name}' operation function does not preserve shape: {fan_in_shape} -> {self.operation(jnp.zeros_like(fan_in_shape)).shape}")
    
    if len(self.operation_object.parameters) > 0:
      raise SystemError(f"Operation function has parameters: {self.operation_object.paremeters}")
    
    return {}, fan_in_shape
  
  def forward(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    return self.function.forward(inputs), inputs  # WS is just inputs for backprop
  
  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    return self.function.backward(error, weighted_sums), {}

class Dropout(Layer):
  layer_seed = 0
  training_only=True
  def __init__(self, rate:float, mode:str, name:str="", *args, **kwargs):
    """
    Dropout
    -----
      A layer that randomly sets a fraction rate of its inputs to zero during training time
      and scales up its functions by a factor of 1/(1-rate) during test time.
    -----
    Args
    -----
    - rate            (float)  : the fraction of the input units to drop
    - mode            (string) : the mode of dropout, either 'random' where each unit is dropped independently of eachother in accordance to the dropout rate or 'fixed' where a fixed number of units are dropped in accordance to the dropout rate
    - (Optional) name (string) : the name of the layer
    """
    if not (0.0 <= rate < 1.0):
      raise ValueError("Dropout rate must be in the range [0.0, 1.0)")
    if mode.lower() not in ('random', 'fixed'):
      raise ValueError("Dropout mode must be 'random' or 'fixed'")
    
    self.mode = mode.lower() 
    self.rate = rate
    self.name = name
  
  def calibrate(self, fan_in_shape:tuple[int,...], fan_out_shape:tuple[int,...]) -> tuple[dict, tuple[int,...]]:
    return {}, fan_in_shape
  
  def forward(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    if self.rate == 0.0:
      return inputs, jnp.ones_like(inputs)
    
    if self.mode == 'random':
      mask = jax.random.bernoulli(jax.random.PRNGKey(random.randint(1,1000)), p=1.0 - self.rate, shape=inputs.shape)

    else:
      drop_rate = int(jnp.ones((inputs.shape)).flatten().shape[0] * self.rate)
      ones = jnp.ones((inputs.shape)).flatten()[0:drop_rate]
      zeros = jnp.zeros((inputs.shape)).flatten()[drop_rate:]

      mask = jax.random.permutation(jax.random.PRNGKey(random.randint(1,1000)), jnp.concatenate((ones, zeros))).reshape(inputs.shape)

    return inputs * mask, mask
  
  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    if self.rate == 0.0:
      return error, {}
    
    return error * weighted_sums / (1.0 - self.rate), {}

class Reshape(Layer):
  layer_seed = 0
  training_only=False
  def __init__(self, target_shape:tuple[int,...], name:str="", *args, **kwargs):
    """
    Reshape
    -----
      A layer that reshapes its input to a target shape without changing the number of elements.
    -----
    Args
    -----
    - target_shape (tuple[int,...]) : the target shape to reshape to
    - (Optional) name (string) : the name of the layer
    """
    self.target_shape = target_shape
    self.name = name
  
  def calibrate(self, fan_in_shape:tuple[int,...], fan_out_shape:tuple[int,...]) -> tuple[dict, tuple[int,...]]:
    input_size = jnp.prod(jnp.array(fan_in_shape[1:]))  # exclude batch dimension
    target_size = jnp.prod(jnp.array(self.target_shape))
    
    if input_size != target_size:
      raise ValueError(f"Reshape layer '{self.name}' cannot reshape from {fan_in_shape} to {self.target_shape} due to size mismatch ({input_size} -> {target_size})")
    
    return {}, self.target_shape
  
  def forward(self, params:dict, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch_size = inputs.shape[0]
    reshaped_output = inputs.reshape((batch_size, *self.target_shape))
    
    return reshaped_output, inputs  # WS is just inputs for backprop
  
  def backward(self, params:dict, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    upstream_gradient = error.reshape(inputs.shape)
    return upstream_gradient, {}
  
