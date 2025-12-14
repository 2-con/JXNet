"""
Losses
=====
  Loss functions are used to compute the loss of a model and its gradients with respect to the loss. This module also contain a loss calculator used in 
  Staticnet that is not meant to be passed to a model as a loss function.

Provides:
- Loss
  - The base class all JXNet losses must inherit and follow.
    Contains scaffolding for custom and built-in layers.

- Mean Squared Error
- Root Mean Squared Error
- Mean Absolute Error
- Total Squared Error
- Total Absolute Error
- Categorical Cross Entropy
  - Applies the softmax function before computing the cross-entropy loss to simplify the jacobian
- Sparse Categorical Cross Entropy
  - Applies the softmax function before computing the cross-entropy loss to simplify the jacobian
- Binary Cross Entropy
  - Applies the sigmoid function before computing the cross-entropy loss to imporve numerical stability if specified to do so, defaults to false.
"""

import jax.numpy as jnp
from abc import ABC, abstractmethod
import jax

class Loss(ABC):
  """
  Base class for all Loss functions
  
  A Loss class must implement the following methods:
  - 'forward' : the forward pass of the loss function discounting regularization
    - Args:
      - y_true (jnp.ndarray): the true labels for the batch
      - y_pred (jnp.ndarray): the predicted labels for the batch
    - Returns:
      - jnp.float32: the loss value
  
  - 'backward' : the backward pass of the loss function for the initial error
    - Args:
      - y_true (jnp.ndarray): the true labels for the batch
      - y_pred (jnp.ndarray): the predicted labels for the batch
    - Returns:
      - jnp.ndarray: the initial error that will be fed into the backpropagation process

  ### Examples
    Here is an example for the Mean Squared Error loss function
  
  ```
   class Mean_Squared_Error(Loss):
    @staticmethod
    def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
      return jnp.mean(jnp.square(y_true - y_pred)) / 2.0

    @staticmethod
    def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
      return (y_pred - y_true) / y_true.size
  ```
  """
  @abstractmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    The forward pass of the loss function are used to compute the loss, this internal method should be made a static method
    since the loss function is stateless
    
    - 'forward' : the forward pass of the loss function discounting regularization
    - Args:
      - y_true (jnp.ndarray): the true labels for the batch
      - y_pred (jnp.ndarray): the predicted labels for the batch
    - Returns:
      - jnp.float32: the loss value
    """
    pass

  @abstractmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    The backward pass of the loss function are used to compute the initial error, this internal method should be made a static method
    since the loss function is stateless. StandardNet/NetLab does not check for NaN or mathematical errors.
    
    - 'backward' : the backward pass of the loss function for the initial error
    - Args:
      - y_true (jnp.ndarray): the true labels for the batch
      - y_pred (jnp.ndarray): the predicted labels for the batch
    - Returns:
      - jnp.ndarray: the initial error that will be fed into the backpropagation process
    """
    pass

class Loss_calculator:
  """
  Loss class for calculating the loss as well as regularized gradients.
  """
  @staticmethod
  def forward_loss(y_true:jnp.ndarray, y_pred:jnp.ndarray, loss, regularization_lambda:float, regularization_type:str, parameters_pytree:dict):
    """
    Forward Loss
    -----
      Calculate the total loss for a given batch of y_true and y_pred. This includes both the empirical loss and regularization penalty.
    -----
    Args
    -----
    - y_true (jnp.ndarray) : the true labels for the batch
    - y_pred (jnp.ndarray) : the predicted labels for the batch
    - loss_class (core.flash.losses object) : the class of the loss function to use
    - regularization_lambda (float) : the regularization strength
    - regularization_type (str) : the type of regularization to use ("L1" or "L2")
    - parameters_pytree (dict) : a pytree of parameters for the model
    
    Returns:
    - float : the total loss for the batch
    """
    assert y_true.ndim == y_pred.ndim, f"y_true and y_pred must have the same number of dimensions, got {y_true.ndim} and {y_pred.ndim} respectively."
    
    emperical_loss = loss.forward(y_true, y_pred)
    
    regularization_penalty = 0.0
    
    for _, parameters in parameters_pytree.items():
      for param_name, param_value in parameters.items():
        if param_name in ('bias', 'biases'):
          continue
        
        if regularization_type == "L2":
          regularization_penalty += jnp.sum(jnp.square(param_value))
        elif regularization_type == "L1":
          regularization_penalty += jnp.sum(jnp.abs(param_value))
        else:
          continue
        
    return emperical_loss + regularization_lambda * regularization_penalty
  
  @staticmethod
  def regressor_forward_loss(y_true:jnp.ndarray, y_pred:jnp.ndarray, loss, regularization_lambda:float, regularization_type:str, parameters:dict):
    """
    Forward Loss
    -----
      Calculate the total loss for a given batch of y_true and y_pred. This includes both the empirical loss and regularization penalty.
      However, this function is specifically for regressors.
    -----
    Args
    -----
    - y_true (jnp.ndarray) : the true labels for the batch
    - y_pred (jnp.ndarray) : the predicted labels for the batch
    - loss_class (core.flash.losses object) : the class of the loss function to use
    - regularization_lambda (float) : the regularization strength
    - regularization_type (str) : the type of regularization to use ("L1" or "L2")
    - parameters_pytree (dict) : a pytree of parameters for the model
    
    Returns:
    - float : the total loss for the batch
    """
    assert y_true.ndim == y_pred.ndim, f"y_true and y_pred must have the same number of dimensions, got {y_true.ndim} and {y_pred.ndim} respectively."
    emperical_loss = loss.forward(y_true, y_pred)
    
    regularization_penalty = 0.0
    
    for param_name, param_value in parameters.items():
      if param_name in ('bias', 'biases'):
        continue
      
      if regularization_type == "L2":
        regularization_penalty += jnp.sum(jnp.square(param_value))
      elif regularization_type == "L1":
        regularization_penalty += jnp.sum(jnp.abs(param_value))
      else:
        continue
        
    return emperical_loss + regularization_lambda * regularization_penalty
  
  @staticmethod
  def regularize_grad(layer_params:dict, gradients:jnp.ndarray, regularization_lambda, regularization_type, ignore_list=['bias', 'biases']):
    """
    Regularize Gradient
    -----
      Modify the gradients of the parameters according to the regularization type and strength.
    -----
    Args
    -----
    - layer_params (dict) : a dictionary of parameters for the layer
    - gradients (dict) : a dictionary of gradients for the layer
    - regularization_lambda (float) : the regularization strength
    - regularization_type (str) : the type of regularization to use ("L1" or "L2")
    
    Returns
    ----
    - dict : the modified gradients for the layer
    """
    for param_name, param_value in layer_params.items():
      if param_name in ignore_list:
        continue
      
      if regularization_type.lower() == "L2":
        gradients[param_name] += 2 * regularization_lambda * param_value
      
      elif regularization_type.lower() == "L1":
        gradients[param_name] += regularization_lambda * jnp.sign(param_value)
      
      else:
        continue
      
    return gradients

##########################################################################################################
#                                            Built-in Contents                                           #
##########################################################################################################

class Mean_Squared_Error(Loss):
  """
  Mean Squared Error
  -----
    Calculate the mean squared error between two JAX NumPy arrays.
  """
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.square(y_true - y_pred)) / 2.0
  
  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return (y_pred - y_true) / y_true.size

class Root_Mean_Squared_Error(Loss):
  """
  Root Mean Squared Error
  -----
    Calculate the root mean squared error between two JAX NumPy arrays.
  """
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(Mean_Squared_Error.forward(y_true, y_pred))
  
  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    mse_grad = Mean_Squared_Error.backward(y_true, y_pred)
    mse = Mean_Squared_Error.forward(y_true, y_pred)
    return mse_grad / (2 * jnp.sqrt(mse))

class Mean_Absolute_Error(Loss):
  """
  Mean Absolute Error (L1 Loss)
  -----
    Calculate the mean absolute error between two JAX NumPy arrays.
    This is also refered to as L1 loss.
  """
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(y_true - y_pred))

  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(y_pred - y_true) / y_true.size

class Total_Squared_Error(Loss):
  """
  Total Squared Error
  -----
    Calculate the total squared error between two JAX NumPy arrays.
  """
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.square(y_true - y_pred)) / 2.0

  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return (y_pred - y_true)

class Total_Absolute_Error(Loss):
  """
  Total Absolute Error
  -----
    Calculate the total absolute error between two JAX NumPy arrays.
  """
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.abs(y_true - y_pred))
  
  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(y_pred - y_true)

class Categorical_Crossentropy(Loss):
  """
  Categorical Crossentropy
  -----
    Calculate the categorical crossentropy between two JAX NumPy arrays.
    
    The softmax function is applied to the predicted values and then the loss is calculated to simplify the derivative jacobian calculation to
    propagate backwards; it is encouraged to pass a JXNet softmax activation at the outer layer to explicitly apply softmax after training when the 
    cancelling effect of softmax is no longer needed.
  """
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    softmax_pred = jax.nn.softmax(y_pred)
    epsilon = 1e-5
    clipped_softmax_pred = jnp.clip(softmax_pred, epsilon, 1.0 - epsilon)
    return -jnp.mean(jnp.sum(y_true * jnp.log(clipped_softmax_pred), axis=-1))
  
  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return y_pred - y_true

class Sparse_Categorical_Crossentropy(Loss):
  """
  Sparse Categorical Crossentropy
  -----
    Calculate the sparse categorical crossentropy between two JAX NumPy arrays where the true labels are integers which are convertered to one-hot.
    
    The softmax function  is applied to the predicted values and then the loss is calculated to simplify the derivative jacobian calculation to
    propagate backwards; it is encouraged to pass a JXNet softmax activation at the outer layer to explicitly apply softmax after training when the 
    cancelling effect of softmax is no longer needed.
  """
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    softmax_pred = jax.nn.softmax(y_pred)
    epsilon = 1e-5
    clipped_softmax_pred = jnp.clip(softmax_pred, epsilon, 1.0 - epsilon)
    true_class_probabilities = jnp.take_along_axis(clipped_softmax_pred, y_true[:, None], axis=-1).squeeze(-1)
    return -jnp.mean(jnp.log(true_class_probabilities))
  
  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    epsilon = 1e-5
    y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
    num_classes = y_pred.shape[-1]
    
    one_hot_labels = jax.nn.one_hot(y_true, num_classes=num_classes)
    return y_pred - one_hot_labels

class Binary_Crossentropy(Loss):
  """
  Binary Crossentropy
  -----
    Calculate the binary crossentropy between two JAX NumPy arrays.
    
  Args
  -----
  - from_logits (bool): whether to apply the sigmoid function to the predicted values during the forward pass, defaults to False
  """
  def __init__(self, from_logits=False):
    self.from_logits = from_logits
  
  def forward(self, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    if self.from_logits:
      y_pred = jax.nn.sigmoid(y_pred)
    
    epsilon = 1e-5
    y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
    return -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

  def backward(self, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    if self.from_logits:
      return jax.nn.sigmoid(y_pred) - y_true
    
    epsilon = 1e-5
    y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
    return - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred))

