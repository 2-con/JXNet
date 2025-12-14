"""
Optimizer
=====
  Optimizers are algorithms that update the parameters of the model based on the gradients of the loss function along with optimizer-spesific hyperparameters
  and information.

Provides:

- Optimizer
  - Base class for all optimizers.
    Contains scaffolding for custom and built-in layers.

- AMSGrad
- Default (SGD/Gradient Descent)
- GradClip (Gradient Clipping)
- SGND (Sign Gradient Descent)
- Momentum
- RMSprop (Root Mean Square Propagation)
- Adagrad
- Novograd
- Adam
- Adadelta
- Adamax
- Rprop (Resillient Propagation)
"""

import jax.numpy as jnp
from abc import ABC, abstractmethod

class Optimizer(ABC):
  """
  Base class for all Optimizers
  
  An Optimizer must implement the following methods:
  - '__init__' : a way to contain the optimizer hyperparameters to be considered later, these have to be static
    - Args:
      - *args : The hyperparameters of the optimizer
      - **kwargs : The hyperparameters of the optimizer
    - Returns:
      - None
  
  - 'update' : a method that contains the logic for updating the parameters
    - Args:
      - lr (float) : The learning rate
      - param (jnp.ndarray) : The current parameters
      - gradient (jnp.ndarray) : The gradient of the loss with respect to the parameters
      - opt_state (tuple) : The state of the optimizer
    - Returns:
      - jnp.ndarray : The updated parameters
      - tuple[gradient, ...] : The updated state of the optimizer (new opt_state). must contain the gradients as the first element
    
  - 'initialize' : a method that returns the initial state of the optimizer
    - Args:
      - param_shape (tuple) : The shape of the parameters
      - param_dtype (jnp.dtype) : The dtype of the parameters
    - Returns:
      - tuple : The initial state of the optimizer
  
  ### Example
    Here is an example of a simple custom optimizer that implements gradient descent
  ```
  class Default(Optimizer):
    def __init__(self, learning_rate, *args, **kwargs):
      self.learning_rate = learning_rate
    
    def update(self, param, gradient, opt_state, **kwargs):
      new_param = param - self.learning_rate * gradient
      return new_param, (gradient,)
    
    @staticmethod
    def initialize(param_shape, param_dtype):
      return (jnp.zeros(param_shape, dtype=param_dtype),)
  ```
  """
  
  @abstractmethod
  def __init__(self, *args, **kwargs):
    """
    __init__ method allows the optimizer object to store static hyperparameters
    
    - Args:
      - *args : The hyperparameters of the optimizer
      - **kwargs : The hyperparameters of the optimizer
    - Returns:
      - None
    """
    pass
  
  @abstractmethod
  def update(self, param, gradient, opt_state, **kwargs):
    """
    the update method contains the logic for updating the parameters. This is a required method
    
    - Args:
      - lr (float) : The learning rate
      - param (jnp.ndarray) : The current parameters
      - gradient (jnp.ndarray) : The gradient of the loss with respect to the parameters
      - opt_state (tuple) : The state of the optimizer
    - Returns:
      - jnp.ndarray : The updated parameters
      - tuple[gradient, ...] : The updated state of the optimizer (new opt_state). must contain the gradients as the first element
    """
    pass

  @abstractmethod
  def initialize(self, param_shape, param_dtype):
    """
    This method is used to initialize the values of the optimizer state (opt_state). not all optimizers take in completely empty opt_state
    
    This method should be labeled as static if possible
    
    - Args:
      - param_shape (tuple) : The shape of the parameters
      - param_dtype (jnp.dtype) : The dtype of the parameters
    - Returns:
      - tuple : The initial state of the optimizer, design the optimizer around this since pytrees are used to map it per-layer
    """
    pass

##########################################################################################################
#                                            Built-in Contents                                           #
##########################################################################################################
import jax
class AMSgrad(Optimizer):
  """
  AMSgrad (Adaptive Moment Square Gradient)
  -----
    An improvement over Adam that addresses a potential non-convergence issue by maintaining the maximum of all past second moment vectors (V_hat). 
    It uses this maximum V_hat to normalize the update, ensuring the effective learning rate is non-increasing
  """
  def __init__(self, learning_rate, alpha=0.9, beta=0.999, epsilon=1e-3):
    self.learning_rate = learning_rate
    self.alpha = alpha
    self.beta = beta
    self.epsilon = epsilon
  
  def update(self, param, gradient, opt_state, **kwargs):
    alpha = self.alpha
    beta = self.beta
    epsilon = self.epsilon
    timestep = kwargs.get('timestep', 1)

    # Access state components by index
    m = opt_state["m"]          # m
    v = opt_state["v"]          # v
    v_hat_max = opt_state["v hat max"]  # v_hat_max

    m_new = (alpha * m) + ((1 - alpha) * gradient)
    v_new = (beta * v) + ((1 - beta) * jnp.square(gradient))

    v_hat_max_new = jnp.maximum(v_hat_max, v_new)

    M_hat = m_new / (1 - alpha**timestep)
    
    new_param = param - (self.learning_rate / (jnp.sqrt(v_hat_max_new) + epsilon)) * M_hat

    return new_param, {
      "gradient": gradient,
      "m": m_new,
      "v": v_new,
      "v hat max": v_hat_max_new
    }
  
  @staticmethod
  def initialize(param_shape, param_dtype):
    return {
        "gradient": jnp.zeros(param_shape, dtype=param_dtype),
        "m": jnp.zeros(param_shape, dtype=param_dtype),
        "v": jnp.zeros(param_shape, dtype=param_dtype),
        "v hat max": jnp.zeros(param_shape, dtype=param_dtype)
    }

class Default(Optimizer):
  """
  Default Gradient Descent
  -----
    Iterative first-order optimization that updates parameters in the direction opposite to the gradient of the loss function. 
    When using the entire dataset, it is Gradient Descent (GD); when using a single sample or mini-batch, it is Stochastic Gradient Descent (SGD).
  """
  def __init__(self, learning_rate, *args, **kwargs):
    self.learning_rate = learning_rate
  
  def update(self, param, gradient, opt_state, **kwargs):
    new_param = param - self.learning_rate * gradient
    return new_param, {"gradient": gradient}
  
  @staticmethod
  def initialize(param_shape, param_dtype):
    return {"gradient": jnp.zeros(param_shape, dtype=param_dtype)}

class Gradclip(Optimizer):
  """
  Gradient Clipping
  -----
  A regularization technique that prevents exploding gradients by re-scaling (clipping) the gradient vector's magnitude (norm) or individual component 
  values if they exceed a predefined threshold. It is critical for Recurrent Neural Networks (RNNs).
  """
  def __init__(self, learning_rate, minimum=-1e-4, maximum=1e-4):
    self.learning_rate = learning_rate
    self.minimum = minimum
    self.maximum = maximum
  def update(self, param, gradient, opt_state, **kwargs):
    
    minimum = self.minimum
    maximum = self.maximum

    new_param = param - self.learning_rate * jnp.clip(gradient, minimum, maximum)
    return new_param, {"gradient": gradient}
  
  @staticmethod
  def initialize(param_shape, param_dtype):
    return {"gradient": jnp.zeros(param_shape, dtype=param_dtype)}

class SGND(Optimizer):
  """
  Sign Gradient Descent
  -----
    An optimization method that discards the magnitude of the gradient and updates parameters only based on the sign (+1 or -1) of the partial derivative.
    The step size is constant. This provides robustness to large, noisy gradients.
  """
  def __init__(self, learning_rate, *args, **kwargs):
    self.learning_rate = learning_rate
  
  def update(self, param, gradient, opt_state, **kwargs):
    new_param = param - self.learning_rate * jnp.sign(gradient)
    return new_param, {"gradient": gradient}
  
  @staticmethod
  def initialize(param_shape, param_dtype):
    return {"gradient": jnp.zeros(param_shape, dtype=param_dtype)}

class Momentum(Optimizer):
  """
  Momentum
  -----
    Accelerates SGD in relevant directions and dampens oscillation. It accumulates an exponentially weighted moving average of past gradients (velocity) 
    and uses this velocity to push the parameters through the loss landscape
  """
  def __init__(self, learning_rate, alpha=0.9, *args, **kwargs):
    self.learning_rate = learning_rate
    self.alpha = alpha
  
  def update(self, param, gradient, opt_state, **kwargs):
    alpha = self.alpha
    velocity = opt_state["velocity"]
    
    new_velocity = (alpha * velocity) + (self.learning_rate * gradient)
    new_param = param - new_velocity
    
    return new_param, {
      "gradient": gradient,
      "velocity": new_velocity
    }
  
  @staticmethod
  def initialize(param_shape, param_dtype):
    return {
      "gradient": jnp.zeros(param_shape, dtype=param_dtype),
      "velocity": jnp.zeros(param_shape, dtype=param_dtype)
    }
  
class RMSprop(Optimizer):
  """
  RMSprop (Root Mean Square Propagation)
  -----
    Adaptive learning rate optimizer that addresses AdaGrad's diminishing rate by using an exponentially decaying average (EMA) of the squared gradients, 
    rather than the cumulative sum. It normalizes parameter updates by the root mean square of these recent squared gradients
  """
  def __init__(self, learning_rate, alpha=0.9, epsilon=1e-3, *args, **kwargs):
    self.learning_rate = learning_rate
    self.alpha = alpha
    self.epsilon = epsilon
  
  def update(self, param, gradient, opt_state, **kwargs):
    alpha = self.alpha
    epsilon = self.epsilon
    # Access state components by index
    avg_sq_grad = opt_state["avg squared grad"] # avg_sq_grad
    
    avg_sq_grad_new = (alpha * avg_sq_grad) + ((1 - alpha) * jnp.square(gradient))
    RMS_gradient = jnp.sqrt(avg_sq_grad_new + epsilon)
    new_param = param - self.learning_rate * (gradient / RMS_gradient)
    
    return new_param, {
      "gradient": gradient,
      "avg squared grad": avg_sq_grad_new
    }
  
  @staticmethod
  def initialize(param_shape, param_dtype):
    return {
      "gradient": jnp.zeros(param_shape, dtype=param_dtype),
      "avg squared grad": jnp.zeros(param_shape, dtype=param_dtype)
    }

class Adagrad(Optimizer):
  """
  AdaGrad (Adaptive Gradient)
  -----
    Adaptive learning rate optimizer that scales the learning rate inversely proportional to the square root of the sum of all previous squared gradients 
    for each parameter. It is effective for sparse data but causes the learning rate to diminish aggressively over time
  """
  def __init__(self, learning_rate, epsilon=1e-3, *args, **kwargs):
    self.learning_rate = learning_rate
    self.epsilon = epsilon
  
  def update(self, param, gradient, opt_state, **kwargs):
    epsilon = self.epsilon
    # Access state components by index
    sum_sq_grad = opt_state["sum squared grad"] # sum_sq_grad
    
    sum_sq_grad_new = sum_sq_grad + jnp.square(gradient)
    new_param = param - (self.learning_rate / (jnp.sqrt(sum_sq_grad_new) + epsilon)) * gradient
    
    return new_param, {
      "gradient": gradient,
      "sum squared grad": sum_sq_grad_new
    }
  
  @staticmethod
  def initialize(param_shape, param_dtype):
    return {
      "gradient": jnp.zeros(param_shape, dtype=param_dtype),
      "sum squared grad": jnp.zeros(param_shape, dtype=param_dtype),
    }

class Novograd(Optimizer):
  """
  Novograd
  -----
    A first-order adaptive gradient method that computes and normalizes the second moment (squared gradient) on a layer-wise basis, rather than per-parameter.
    It is robust to large batch sizes and often performs well without learning rate warmup
  """
  def __init__(self, learning_rate, alpha=0.9, beta=0.999, epsilon=1e-3, *args, **kwargs):
    self.learning_rate = learning_rate
    self.alpha = alpha
    self.beta = beta
    self.epsilon = epsilon
  
  def update(self, param, gradient, opt_state, **kwargs):
    alpha = self.alpha
    beta = self.beta
    epsilon = self.epsilon
    timestep = kwargs.get('timestep', 1)
    # Access state components by index
    m = opt_state["m"] # m
    v = opt_state["v"] # v
  
    normalized_gradient = gradient / (jnp.abs(gradient) + epsilon)
    m_new = (alpha * m) + ((1 - alpha) * normalized_gradient)
    v_new = (beta * v) + ((1 - beta) * jnp.square(gradient))
    
    M_hat = m_new / (1 - alpha**timestep)
    V_hat = v_new / (1 - beta**timestep)
    new_param = param - ((M_hat * self.learning_rate) / (jnp.sqrt(V_hat) + epsilon))
    
    return new_param, (gradient, m_new, v_new)
  
  @staticmethod
  def initialize(param_shape, param_dtype):
    return {
      "gradient": jnp.zeros(param_shape, dtype=param_dtype),
      "m": jnp.zeros(param_shape, dtype=param_dtype),
      "v": jnp.zeros(param_shape, dtype=param_dtype)
    }

class Adam(Optimizer):
  """
  Adam (Adaptive Moment Estimation)
  -----
    Combines the benefits of Momentum (using EMA of gradients, the first moment) and RMSProp 
    (using EMA of squared gradients, the second moment) to compute individual adaptive learning rates for each parameter. Includes a bias-correction term
  """
  def __init__(self, learning_rate, alpha=0.9, beta=0.999, epsilon=1e-3):
    self.learning_rate = learning_rate
    self.alpha = alpha
    self.beta = beta
    self.epsilon = epsilon
  
  def update(self, param, gradient, opt_state, **kwargs):
    alpha = self.alpha
    beta = self.beta
    epsilon = self.epsilon
    timestep = kwargs.get('timestep', 1)
    
    # Access state components by index
    m = opt_state["m"] # m
    v = opt_state["v"] # v
    
    m_new = (alpha * m) + ((1 - alpha) * gradient)
    v_new = (beta * v) + ((1 - beta) * jnp.square(gradient))
    
    M_hat = m_new / (1 - alpha**timestep)
    V_hat = v_new / (1 - beta**timestep)
    new_param = param - ((M_hat * self.learning_rate) / (jnp.sqrt(V_hat) + epsilon))
    
    return new_param, {
      "gradient": gradient,
      "m": m_new,
      "v": v_new
    }
  
  @staticmethod
  def initialize(param_shape, param_dtype):
    return {
      "gradient": jnp.zeros(param_shape, dtype=param_dtype),
      "m": jnp.zeros(param_shape, dtype=param_dtype),
      "v": jnp.zeros(param_shape, dtype=param_dtype)
    }

class Adadelta(Optimizer):
  """
  AdaDelta
  -----
    An extension of RMSProp that eliminates the global learning rate hyperparameter. It maintains two moving averages: one for the squared gradients and 
    one for the squared parameter updates, ensuring the update step is dimensionally consistent (unit-aware)
  """
  def __init__(self, alpha=0.9, epsilon=1e-3, *args, **kwargs):
    self.alpha = alpha
    self.epsilon = epsilon
  
  def update(self, param, gradient, opt_state, **kwargs):
    alpha = self.alpha
    epsilon = self.epsilon
    avg_sq_grad = opt_state["avg squared grad"]
    avg_sq_delta = opt_state["avg squared delta"]
    
    avg_sq_grad_new = (alpha * avg_sq_grad) + ((1 - alpha) * jnp.square(gradient))
    RMS_gradient = jnp.sqrt(avg_sq_grad_new + epsilon)
    RMS_delta = jnp.sqrt(avg_sq_delta + epsilon)
    
    delta = (RMS_delta / RMS_gradient) * gradient
    avg_sq_delta_new = (alpha * avg_sq_delta) + ((1 - alpha) * jnp.square(delta))
    new_param = param - delta
    
    return new_param, {
      "gradient": gradient,
      "avg squared grad": avg_sq_grad_new,
      "avg squared delta": avg_sq_delta_new
    }

  @staticmethod
  def initialize(param_shape, param_dtype):
    return {
        "gradient": jnp.zeros(param_shape, dtype=param_dtype),
        "avg squared grad": jnp.zeros(param_shape, dtype=param_dtype),
        "avg squared delta": jnp.zeros(param_shape, dtype=param_dtype)
    }

class Adamax(Optimizer):
  """
  AdaMax
  -----
    A variant of Adam that simplifies the computation of the second moment (V) by using the L-infinity norm (max norm) of past gradients rather than 
    the L2 norm. It can be more numerically stable for large, sparse gradients.
  """
  def __init__(self, learning_rate, alpha=0.9, beta=0.999, epsilon=1e-3, *args, **kwargs):
    self.learning_rate = learning_rate
    self.alpha = alpha
    self.beta = beta
    self.epsilon = epsilon
  
  def update(self, param, gradient, opt_state, **kwargs):
    alpha = self.alpha
    beta = self.beta
    epsilon = self.epsilon
    m = opt_state["m"]
    u_inf = opt_state["u inf"]
    
    m_new = (alpha * m) + ((1 - alpha) * gradient)
    u_inf_new = jnp.maximum(beta * u_inf, jnp.abs(gradient))
    M_hat = m_new / (1 - alpha) # No timestep needed for bias correction in Adamax m
    new_param = param - (self.learning_rate * M_hat / (u_inf_new + epsilon))
    
    return new_param, {
      "gradient": gradient,
      "m": m_new,
      "u inf": u_inf_new
    }

  @staticmethod
  def initialize(param_shape, param_dtype):
    return {
      "gradient": jnp.zeros(param_shape, dtype=param_dtype),
      "m": jnp.zeros(param_shape, dtype=param_dtype),
      "u inf": jnp.zeros(param_shape, dtype=param_dtype)
    }

class Rprop(Optimizer):
  """
  Rprop (Resilient Backpropagation)
  -----
    A batch-mode optimization algorithm that uses only the sign of the partial derivative to determine the direction of the weight update. 
    It maintains a separate adaptive step size for each weight, which increases if the sign is consistent and decreases if the sign flips
  """
  def __init__(self, alpha=1.1, beta=0.5, min_step=1e-6, max_step=1e-2, *args, **kwargs):
    self.alpha = alpha
    self.beta = beta
    self.min_step = min_step
    self.max_step = max_step
  
  def update(self, param, gradient, opt_state, **kwargs):
    alpha = self.alpha # Increase factor
    beta = self.beta  # Decrease factor
    min_step = self.min_step # Minimum step size
    max_step = self.max_step # Maximum step size
    prev_grad = opt_state["prev grad"]
    step_size = opt_state["step size"]

    signs_agree = jnp.sign(prev_grad) * jnp.sign(gradient) > 0.0
    
    # Increase or decrease step size
    new_step_size = jnp.where(
      signs_agree,
      step_size * alpha,
      step_size * beta
    )
    
    # Clip step size
    new_step_size = jnp.clip(new_step_size, min_step, max_step)
    
    # If signs disagree, the previous gradient is set to zero (for next iteration)
    new_prev_grad = jnp.where(signs_agree, gradient, jnp.zeros_like(gradient))
    update_delta = jnp.where(signs_agree, new_step_size * jnp.sign(gradient), jnp.zeros_like(gradient))
    new_param = param - update_delta
    
    return new_param, {
      "gradient": gradient,
      "prev grad": new_prev_grad,
      "step size": new_step_size
    }

  @staticmethod
  def initialize(param_shape, param_dtype):
    return {
      "gradient": jnp.zeros(param_shape, dtype=param_dtype),
      "prev grad": jnp.zeros(param_shape, dtype=param_dtype),
      "step size": jnp.full(param_shape, 0.01, dtype=param_dtype)
    }

