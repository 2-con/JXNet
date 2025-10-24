import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
      
      if regularization_type == "L2":
        gradients[param_name] += 2 * regularization_lambda * param_value
      
      elif regularization_type == "L1":
        gradients[param_name] += regularization_lambda * jnp.sign(param_value)
      
      else:
        continue
      
    return gradients

class Mean_Squared_Error(Loss):
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.square(y_true - y_pred))
  
  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return 2 * (y_pred - y_true) / y_true.size

class Root_Mean_Squared_Error(Loss):
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(Mean_Squared_Error.forward(y_true, y_pred))
  
  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    mse_grad = Mean_Squared_Error.backward(y_true, y_pred)
    mse = Mean_Squared_Error.forward(y_true, y_pred)
    return mse_grad / (2 * jnp.sqrt(mse))

class Mean_Absolute_Error(Loss):
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(y_true - y_pred))

  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.sign(y_pred - y_true))

class Total_Squared_Error(Loss):
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.square(y_true - y_pred)) / 2.0

  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return (y_pred - y_true)

class Total_Absolute_Error(Loss):
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.abs(y_true - y_pred))
  
  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(y_pred - y_true)

class L1_Loss(Loss):
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(y_pred - y_true))

  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.sign(y_pred - y_true))

# classification loss functions
class Categorical_Crossentropy(Loss):
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
    return -jnp.mean(jnp.sum(y_true * jnp.log(y_pred), axis=-1))

  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
    return -y_true / y_pred

class Sparse_Categorical_Crossentropy(Loss):
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
    true_class_probabilities = jnp.take_along_axis(y_pred, y_true[:, None], axis=-1).squeeze(-1)
    return -jnp.mean(jnp.log(true_class_probabilities))
  
  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
    num_classes = y_pred.shape[-1]
    
    one_hot_labels = jax.nn.one_hot(y_true, num_classes=num_classes)
    
    return -(one_hot_labels / y_pred)

class Binary_Crossentropy(Loss):
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
    return -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
    return - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
