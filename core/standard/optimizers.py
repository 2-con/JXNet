import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
  def update(self, lr, param, gradient, opt_state, **kwargs):
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

class AMSgrad(Optimizer):
  def __init__(self, alpha=0.9, beta=0.999, epsilon=1e-8):
    self.alpha = alpha
    self.beta = beta
    self.epsilon = epsilon
  
  def update(self, lr, param, gradient, opt_state, **kwargs):
    alpha = self.alpha
    beta = self.beta
    epsilon = self.epsilon
    timestep = kwargs.get('timestep', 1)

    # Access state components by index
    m = opt_state[1]          # m
    v = opt_state[2]          # v
    v_hat_max = opt_state[3]  # v_hat_max

    m_new = (alpha * m) + ((1 - alpha) * gradient)
    v_new = (beta * v) + ((1 - beta) * jnp.square(gradient))

    v_hat_max_new = jnp.maximum(v_hat_max, v_new)

    M_hat = m_new / (1 - alpha**timestep)
    
    new_param = param - (lr / (jnp.sqrt(v_hat_max_new) + epsilon)) * M_hat

    # Return updated state components as a tuple
    new_opt_state = (gradient, m_new, v_new, v_hat_max_new)
    return new_param, new_opt_state
  
  @staticmethod
  def initialize(param_shape, param_dtype):
    return (
        jnp.zeros(param_shape, dtype=param_dtype),  # gradient
        jnp.zeros(param_shape, dtype=param_dtype),  # m
        jnp.zeros(param_shape, dtype=param_dtype),  # v
        jnp.zeros(param_shape, dtype=param_dtype)   # v_hat_max
    )

class Default(Optimizer):
  def __init__(self, *args, **kwargs):
    pass
  
  def update(self, lr, param, gradient, opt_state, **kwargs):
    new_param = param - lr * gradient
    return new_param, (gradient,)
  
  @staticmethod
  def initialize(param_shape, param_dtype):
    return (jnp.zeros(param_shape, dtype=param_dtype),)

class Gradclip(Optimizer):
  def __init__(self, minimum=-1e-4, maximum=1e-4):
    self.minimum = minimum
    self.maximum = maximum
  def update(self, lr, param, gradient, opt_state, **kwargs):
    
    minimum = self.minimum
    maximum = self.maximum

    new_param = param - lr * jnp.clip(gradient, minimum, maximum)
    return new_param, (gradient,)
  
  @staticmethod
  def initialize(param_shape, param_dtype):
    return (jnp.zeros(param_shape, dtype=param_dtype),)

class SGND(Optimizer):
  def __init__(self, *args, **kwargs):
    pass
  
  def update(self, lr, param, gradient, opt_state, **kwargs):
    new_param = param - lr * jnp.sign(gradient)
    return new_param, (gradient,)
  
  @staticmethod
  def initialize(param_shape, param_dtype):
    return (jnp.zeros(param_shape, dtype=param_dtype),)

class Momentum(Optimizer):
  def __init__(self, alpha=0.9, *args, **kwargs):
    self.alpha = alpha
    pass
  
  def update(self, lr, param, gradient, opt_state, **kwargs):
    alpha = self.alpha
    velocity = opt_state[0] # velocity
    
    new_velocity = (alpha * velocity) + (lr * gradient)
    new_param = param - new_velocity
    
    return new_param, (gradient, new_velocity)
  
  @staticmethod
  def initialize(param_shape, param_dtype):
    return (jnp.zeros(param_shape, dtype=param_dtype),
            jnp.zeros(param_shape, dtype=param_dtype),)  # velocity
  
class RMSprop(Optimizer):
  def __init__(self, alpha=0.9, epsilon=1e-8, *args, **kwargs):
    self.alpha = alpha
    self.epsilon = epsilon
  
  def update(self, lr, param, gradient, opt_state, **kwargs):
    alpha = self.alpha
    epsilon = self.epsilon
    # Access state components by index
    avg_sq_grad = opt_state[1] # avg_sq_grad
    
    avg_sq_grad_new = (alpha * avg_sq_grad) + ((1 - alpha) * jnp.square(gradient))
    RMS_gradient = jnp.sqrt(avg_sq_grad_new + epsilon)
    new_param = param - lr * (gradient / RMS_gradient)
    
    return new_param, (gradient, avg_sq_grad_new)
  
  @staticmethod
  def initialize(param_shape, param_dtype):
    return (jnp.zeros(param_shape, dtype=param_dtype),
            jnp.zeros(param_shape, dtype=param_dtype),)  # avg_sq_grad

class Adagrad(Optimizer):
  def __init__(self, epsilon=1e-8, *args, **kwargs):
    self.epsilon = epsilon
  
  def update(self, lr, param, gradient, opt_state, **kwargs):
    epsilon = self.epsilon
    # Access state components by index
    sum_sq_grad = opt_state[1] # sum_sq_grad
    
    sum_sq_grad_new = sum_sq_grad + jnp.square(gradient)
    new_param = param - (lr / (jnp.sqrt(sum_sq_grad_new) + epsilon)) * gradient
    
    return new_param, (gradient, sum_sq_grad_new)
  
  @staticmethod
  def initialize(param_shape, param_dtype):
    return (jnp.zeros(param_shape, dtype=param_dtype),
            jnp.zeros(param_shape, dtype=param_dtype),)  # sum_sq_grad

class Novograd(Optimizer):
  def __init__(self, alpha=0.9, beta=0.999, epsilon=1e-8, *args, **kwargs):
    self.alpha = alpha
    self.beta = beta
    self.epsilon = epsilon
  
  def update(self, lr, param, gradient, opt_state, **kwargs):
    alpha = self.alpha
    beta = self.beta
    epsilon = self.epsilon
    timestep = kwargs.get('timestep', 1)
    # Access state components by index
    m = opt_state[1] # m
    v = opt_state[2] # v
  
    normalized_gradient = gradient / (jnp.abs(gradient) + epsilon)
    m_new = (alpha * m) + ((1 - alpha) * normalized_gradient)
    v_new = (beta * v) + ((1 - beta) * jnp.square(gradient))
    
    M_hat = m_new / (1 - alpha**timestep)
    V_hat = v_new / (1 - beta**timestep)
    new_param = param - ((M_hat * lr) / (jnp.sqrt(V_hat) + epsilon))
    
    return new_param, (gradient, m_new, v_new)
  
  @staticmethod
  def initialize(param_shape, param_dtype):
    return (
        jnp.zeros(param_shape, dtype=param_dtype),
        jnp.zeros(param_shape, dtype=param_dtype),  # m
        jnp.zeros(param_shape, dtype=param_dtype)   # v
    )

class Adam(Optimizer):
  def __init__(self, alpha=0.9, beta=0.999, epsilon=1e-8):
    self.alpha = alpha
    self.beta = beta
    self.epsilon = epsilon
  
  def update(self, lr, param, gradient, opt_state, **kwargs):
    alpha = self.alpha
    beta = self.beta
    epsilon = self.epsilon
    timestep = kwargs.get('timestep', 1)
    
    # Access state components by index
    m = opt_state[1] # m
    v = opt_state[2] # v
    
    m_new = (alpha * m) + ((1 - alpha) * gradient)
    v_new = (beta * v) + ((1 - beta) * jnp.square(gradient))
    
    M_hat = m_new / (1 - alpha**timestep)
    V_hat = v_new / (1 - beta**timestep)
    new_param = param - ((M_hat * lr) / (jnp.sqrt(V_hat) + epsilon))
    
    return new_param, (gradient, m_new, v_new)
  
  @staticmethod
  def initialize(param_shape, param_dtype):
    return (
        jnp.zeros(param_shape, dtype=param_dtype),
        jnp.zeros(param_shape, dtype=param_dtype),  # m
        jnp.zeros(param_shape, dtype=param_dtype)   # v
    )

class Adadelta(Optimizer):
  def __init__(self, alpha=0.9, epsilon=1e-8, *args, **kwargs):
    self.alpha = alpha
    self.epsilon = epsilon
  
  def update(self, lr, param, gradient, opt_state, **kwargs):
    alpha = self.alpha
    epsilon = self.epsilon
    # Access state components by index
    avg_sq_grad = opt_state[1] # avg_sq_grad
    avg_sq_delta = opt_state[2] # avg_sq_delta
    
    avg_sq_grad_new = (alpha * avg_sq_grad) + ((1 - alpha) * jnp.square(gradient))
    RMS_gradient = jnp.sqrt(avg_sq_grad_new + epsilon)
    RMS_delta = jnp.sqrt(avg_sq_delta + epsilon)
    
    delta = (RMS_delta / RMS_gradient) * gradient
    avg_sq_delta_new = (alpha * avg_sq_delta) + ((1 - alpha) * jnp.square(delta))
    new_param = param - delta
    
    return new_param, (gradient, avg_sq_grad_new, avg_sq_delta_new)

  @staticmethod
  def initialize(param_shape, param_dtype):
    return (
        jnp.zeros(param_shape, dtype=param_dtype),
        jnp.zeros(param_shape, dtype=param_dtype),  # avg_sq_grad
        jnp.zeros(param_shape, dtype=param_dtype)   # avg_sq_delta
    )

class Adamax(Optimizer):
  def __init__(self, alpha=0.9, beta=0.999, epsilon=1e-8, *args, **kwargs):
    self.alpha = alpha
    self.beta = beta
    self.epsilon = epsilon
  
  def update(self, lr, param, gradient, opt_state, **kwargs):
    alpha = self.alpha
    beta = self.beta
    epsilon = self.epsilon
    # Access state components by index
    m = opt_state[1] # m
    u_inf = opt_state[2] # u_inf
    
    m_new = (alpha * m) + ((1 - alpha) * gradient)
    u_inf_new = jnp.maximum(beta * u_inf, jnp.abs(gradient))
    M_hat = m_new / (1 - alpha) # No timestep needed for bias correction in Adamax m
    new_param = param - (lr * M_hat / (u_inf_new + epsilon))
    
    return new_param, (gradient, m_new, u_inf_new)

  @staticmethod
  def initialize(param_shape, param_dtype):
    return (
        jnp.zeros(param_shape, dtype=param_dtype),
        jnp.zeros(param_shape, dtype=param_dtype),  # m
        jnp.zeros(param_shape, dtype=param_dtype)   # u_inf (max of past gradients)
    )

class Rprop(Optimizer):
  def __init__(self, alpha=1.1, beta=0.5, min_step=1e-6, max_step=50.0, *args, **kwargs):
    self.alpha = alpha
    self.beta = beta
    self.min_step = min_step
    self.max_step = max_step
  
  def update(self, lr, param, gradient, opt_state, **kwargs):
    alpha = self.alpha # Increase factor
    beta = self.beta  # Decrease factor
    min_step = self.min_step # Minimum step size
    max_step = self.max_step # Maximum step size
    
    # Access state components by index
    prev_grad = opt_state[1] # prev_grad
    step_size = opt_state[2] # step_size

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
    # This prevents oscillations
    new_prev_grad = jnp.where(signs_agree, gradient, jnp.zeros_like(gradient))
    
    # Calculate update delta
    update_delta = jnp.where(signs_agree, new_step_size * jnp.sign(gradient), jnp.zeros_like(gradient))
    
    new_param = param - update_delta
    
    return new_param, (gradient, new_prev_grad, new_step_size)

  @staticmethod
  def initialize(param_shape, param_dtype):
    return (
        jnp.zeros(param_shape, dtype=param_dtype),
        jnp.zeros(param_shape, dtype=param_dtype),      # prev_grad
        jnp.full(param_shape, 0.01, dtype=param_dtype)  # step_size (often initialized to a small constant)
    )
