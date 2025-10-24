import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax.numpy as jnp
from abc import ABC, abstractmethod
import jax
from system.defaults import epsilon_default

class Function(ABC):
  """
  Base class for all Function functions. 
  
  Function classes are used to apply a mathematical function to an array and is only used inside a Layer class.
  
  A Function class is required to have the following:
  - `forward` : method for applying the Function function.
    - Args:
      - x (jnp.ndarray): The input array to the Function function.
      - *args: Variable length argument list.  Can be used to pass additional information to the Function function.
      - **kwargs: Arbitrary keyword arguments. Used to pass parameters (if any) to the Function function. 
                Make sure the parameter names match those listed in the 'parameters' attribute.
    - Returns:
      - jnp.ndarray: The output array after applying the Function function, with the same dimensions as the input.
  
  - `backward` : method for computing the gradient of the Function function. 
    - Args:
      - incoming_error (jnp.ndarray): The incoming error signal from the subsequent layer.
      - x (jnp.ndarray): The input to the Function function during the forward pass.  This is needed to compute the gradient.
      - *args: Variable length argument list.  Can be used to pass additional information to the Function function.
      - **kwargs: Arbitrary keyword arguments. Used to pass parameters (if any) to the Function function.
    
    - Returns:
      - dict: A dictionary containing the gradient of the loss with respect to the key (incoming_error * local_gradient).  
            The key are 'x' along with any parametric parameters specified in 'parameters'.
  
  Attributes:
    parameters (list): A list of strings, where each string is the name of a parameter 
                       required by a parametric function. Defaults to an empty list for non-parametric Functions.
  """
  parameters = []
  
  def __init__(self):
    pass
  
  @abstractmethod
  def forward(self, x:jnp.ndarray, *args, **kwargs):
    """
    Forward propagation method: Applies the Function function to the input.
    
    If parametric parameters are defined, then they are passed as keyword arguments so it dosen't matter if its explicitly defined as a perameter
    or if its accessed from the kwargs.
    
    Args:
      x (jnp.ndarray): The input array to the Function function.
      *args: Variable length argument list.  Can be used to pass additional information to the Function function.
      **kwargs: Arbitrary keyword arguments. Used to pass parameters (if any) to the Function function. 
                Make sure the parameter names match those listed in the 'parameters' attribute.
    
    Returns:
      jnp.ndarray: The output array after applying the Function function, with the same dimensions as the input.
      Args:
      incoming_error (jnp.ndarray): The incoming error signal from the subsequent layer.
      x (jnp.ndarray): The input to the Function function during the forward pass.  This is needed to compute the gradient.
      *args: Variable length argument list.  Can be used to pass additional information to the Function function.
      **kwargs: Arbitrary keyword arguments. Used to pass parameters (if any) to the Function function.
    
    Returns:
      dict: A dictionary containing the gradient of the loss with respect to the key (incoming_error * local_gradient).  
            The key are 'x' along with any parametric parameters specified in 'parameters'.
    """
    pass

  def backward(self, incoming_error:jnp.ndarray, x:jnp.ndarray, *args, **kwargs):
    """
    Backward propagation method: Computes the gradient of the Function function with respect to its input.
    
    PyNet will not default to jax.grad to compute the gradient if it is not explicitly defined since some Functions' derivatives
    have to be slight
    
    Args:
      incoming_error (jnp.ndarray): The incoming error signal from the subsequent layer.
      x (jnp.ndarray): The input to the Function function during the forward pass.  This is needed to compute the gradient.
      *args: Variable length argument list.  Can be used to pass additional information to the Function function.
      **kwargs: Arbitrary keyword arguments. Used to pass parameters (if any) to the Function function.
    
    Returns:
      dict: A dictionary containing the gradient of the loss with respect to the key (incoming_error * local_gradient).  
            The key are 'x' along with any parametric parameters specified in 'parameters'.
    """
    pass

# normalization
class Sigmoid(Function):
  
  def forward(self, x, *args, **kwargs):
    return 1.0 / (1.0 + jnp.exp(-x))

  def backward(self, incoming_error, x, *args, **kwargs):
    local_grad = (1.0 / (1.0 + jnp.exp(-x))) * (1 - (1.0 / (1.0 + jnp.exp(-x))))
    return {"x": incoming_error * local_grad} # Outputs dL/dz
  
class Tanh(Function):
  
  def forward(self, x, *args, **kwargs):
    return jnp.tanh(x)

  def backward(self, incoming_error, x, *args, **kwargs):
    local_grad = 1 - jnp.tanh(x)**2
    return {"x": incoming_error * local_grad} # Outputs dL/dz

class Binary_step(Function):
  
  def forward(self, x, *args, **kwargs):
    return jnp.where(x > 0, 1.0, 0.0)

  def backward(self, incoming_error, x, *args, **kwargs):
    return {"x": jnp.zeros_like(x)} # dL/dz is 0

class Softsign(Function):
  
  def forward(self, x, *args, **kwargs):
    return x / (1.0 + jnp.abs(x))

  def backward(self, incoming_error, x, *args, **kwargs):
    local_grad = 1 / (1 + jnp.abs(x))**2
    return {"x": incoming_error * local_grad} # Outputs dL/dz

class Softmax(Function):
  
  def forward(self, x, *args, **kwargs):
    exp_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

  def backward(self, incoming_error, x, *args, **kwargs):
    s = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True)) / jnp.sum(jnp.exp(x - jnp.max(x, axis=-1, keepdims=True)), axis=-1, keepdims=True)
    local_grad = s * (1 - s) 
    return {"x": incoming_error * local_grad}

class ReLU(Function):
  
  def forward(self, x, *args, **kwargs):
    return jnp.maximum(0.0, x)

  def backward(self, incoming_error, x, *args, **kwargs):
    local_grad = jnp.where(x > 0, 1.0, 0.0)
    return {"x": incoming_error * local_grad} # Outputs dL/dz

class Softplus(Function):
  
  def forward(self, x, *args, **kwargs):
    return jnp.log(1.0 + jnp.exp(x))

  def backward(self, incoming_error, x, *args, **kwargs):
    local_grad = 1 / (1 + jnp.exp(-x))
    return {"x": incoming_error * local_grad} # Outputs dL/dz

class Mish(Function):
  
  def forward(self, x, *args, **kwargs):
    return x * jnp.tanh(jnp.log(1.0 + jnp.exp(x)))
  
  # Still requires JAX's grad because the derivative is complex
  def backward(self, incoming_error, x, *args, **kwargs):
    omega = lambda x : x * jnp.tanh(jnp.log(1.0 + jnp.exp(x)))
    raise NotImplemented("jax.grad is broken in core/standard/functions.py")
    local_grad = jax.grad(omega)(x)
    return {"x": incoming_error * local_grad}

class Swish(Function):
  
  def forward(self, x, *args, **kwargs):
    return x * (1.0 / (1.0 + jnp.exp(-x)))

  def backward(self, incoming_error, x, *args, **kwargs):
    s = (1.0 / (1.0 + jnp.exp(-x))) # Sigmoid(x)
    s_prime = s * (1 - s)           # Sigmoid'(x)
    local_grad = s + x * s_prime    # d(x*s)/dx
    return {"x": incoming_error * local_grad}

class Leaky_ReLU(Function):
  
  def forward(self, x, *args, **kwargs):
    return jnp.maximum(0.1 * x, x)

  def backward(self, incoming_error, x, *args, **kwargs):
    local_grad = jnp.where(x > 0, 1.0, 0.1)
    return {"x": incoming_error * local_grad}

class GELU(Function):
  
  def forward(self, x, *args, **kwargs):
    return jax.nn.gelu(x)

  def backward(self, incoming_error, x, *args, **kwargs):
    local_grad = (jax.nn.gelu(x) + (x * (jnp.sqrt(2 / jnp.pi) * jnp.exp(-(x**2) / 2)))) # This is the derivative for the approximate formula.
    return {"x": incoming_error * local_grad}

class Linear(Function):
  
  def forward(self, x, *args, **kwargs):
    return x

  def backward(self, incoming_error, x, *args, **kwargs):
    return {"x": incoming_error}

class ReEU(Function):
  
  def forward(self, x, *args, **kwargs):
    conditions = [x > 10, x < -10]
    choices = [x, 0.0]
    return jnp.select(conditions, choices, default=jnp.minimum(jnp.exp(x), jnp.maximum(1.0, x + 1.0)))

  def backward(self, incoming_error, x, *args, **kwargs):
    conditions = [x > 10, x < -10]
    choices = [1.0, 0.0]
    local_grad = jnp.select(conditions, choices, default=jnp.minimum(jnp.exp(x), 1.0))
    
    return {"x": incoming_error * local_grad}

class ReTanh(Function):
  
  def forward(self, x, *args, **kwargs):
    return x * (jnp.tanh(x + 1.0) + 1.0) / 2.0

  def backward(self, incoming_error, x, *args, **kwargs):
    v = (jnp.tanh(x + 1.0) + 1.0) / 2.0
    v_prime = (1 - jnp.tanh(x + 1.0)**2) / 2.0
    
    local_grad = v + x * v_prime
    
    return {"x": incoming_error * local_grad}

########################################################################################################################
#                                           parametric Functions                                                       #
########################################################################################################################

class ELU(Function):
  parameters = ["alpha"]
  
  def forward(self, x, alpha, *args, **kwargs):
    # Simplified for demonstration; kept original logic as closely as possible
    return jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1.0))

  def backward(self, incoming_error, x, alpha, *args, **kwargs):
    local_grad_x = jnp.where(x > 0, 1.0, alpha * jnp.exp(x))
    local_grad_alpha = jnp.where(x <= 0, (jnp.exp(x) - 1.0), 0.0)
    
    return {
      "x": incoming_error * local_grad_x,
      "alpha": jnp.sum(incoming_error * local_grad_alpha)
    }

class SELU(Function):
  parameters = ["alpha", "beta"]
  # SELU parameters are typically fixed constants
  
  def forward(self, x, alpha, beta, *args, **kwargs):
    return beta * jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1.0))

  def backward(self, incoming_error, x, alpha, beta, *args, **kwargs):
    local_grad_x = beta * jnp.where(x > 0, 1.0, alpha * jnp.exp(x))
    local_grad_alpha = beta * jnp.where(x <= 0, (jnp.exp(x) - 1.0), 0.0)
    
    return {
      "x": incoming_error * local_grad_x,
      "alpha": jnp.sum(incoming_error * local_grad_alpha),
      "beta": jnp.sum(incoming_error * jnp.where(x > 0, x, (alpha * jnp.exp(x) - 1.0)))
    }

class PReLU(Function):
  parameters = ["alpha"]
  
  def forward(self, x, alpha, *args, **kwargs):
    return jnp.maximum(alpha * x, x)

  def backward(self, incoming_error, x, alpha, *args, **kwargs):
    local_grad_x = jnp.where(x > 0, 1.0, alpha)
    local_grad_alpha = jnp.where(x <= 0, x, 0.0)
    
    return {
      "x": incoming_error * local_grad_x,
      "alpha": jnp.sum(incoming_error * local_grad_alpha)
    }

class SiLU(Function):
  parameters = ["alpha"]
  
  def forward(self, x, alpha, *args, **kwargs):
    return x * (1.0 / (1.0 + jnp.exp(-alpha * x)))

  def backward(self, incoming_error, x, alpha, *args, **kwargs):
    s = (1.0 / (1.0 + jnp.exp(-alpha * x))) # Sigmoid(alpha * x)
    s_prime = s * (1 - s)                    # Sigmoid'(alpha * x)
    
    local_grad_x = s + x * alpha * s_prime
    local_grad_alpha = x**2 * s_prime 
    
    return {
      "x": incoming_error * local_grad_x,
      "alpha": jnp.sum(incoming_error * local_grad_alpha)
    }

########################################################################################################################
#                                                     Scalers                                                          #
########################################################################################################################

class Standard_Scaler(Function):
  
  def forward(self, incoming_error, x:jnp.ndarray) -> jnp.ndarray:
    average = jnp.mean(x, axis=0)
    standard_deviation = jnp.std(x, axis=0)

    scaled_x = jnp.where(
      standard_deviation != 0,
      (x - average) / (standard_deviation + epsilon_default),
      0.0
    )
    return scaled_x

  def backward(self, incoming_error, x, *args, **kwargs) -> jnp.ndarray:
    average = jnp.mean(x, axis=0)
    standard_deviation = jnp.std(x, axis=0)

    scaled_x = jnp.where(
      standard_deviation != 0,
      (incoming_error * (standard_deviation + epsilon_default)) + average,
      0.0
    )
    return scaled_x

class Min_Max_Scaler(Function):
  
  def forward(self, x:jnp.ndarray) -> jnp.ndarray:
    max_val = jnp.max(x, axis=0)
    min_val = jnp.min(x, axis=0)
    range_val = max_val - min_val
    
    scaled_x = jnp.where(
      range_val != 0,
      (x - min_val) / (range_val + epsilon_default),
      0.0 
    )
    return scaled_x

  def backward(self, incoming_error, x, *args, **kwargs) -> jnp.ndarray:
    min_val = jnp.min(x, axis=0)
    max_val = jnp.max(x, axis=0)
    range_val = max_val - min_val
    
    scaled_x = jnp.where(
      range_val != 0,
      (incoming_error * (max_val + epsilon_default)) + min_val,
      0.0 
    )
    return scaled_x

class Max_Abs_Scaler(Function):
  
  def forward(self, x:jnp.ndarray) -> jnp.ndarray:
    max_abs_val = jnp.max(jnp.abs(x), axis=0)

    scaled_x = jnp.where(
      max_abs_val != 0,
      x / (max_abs_val + epsilon_default),
      0.0
    )
    return scaled_x

  def backward(self, incoming_error, x, *args, **kwargs) -> jnp.ndarray:
    max_abs_val = jnp.max(jnp.abs(x), axis=0)

    scaled_x = jnp.where(
      max_abs_val != 0,
      incoming_error * (max_abs_val + epsilon_default),
      0.0
    )
    return scaled_x

class Robust_Scaler(Function):
  
  def forward(self, x:jnp.ndarray) -> jnp.ndarray:
    q1 = jnp.quantile(x, 0.25, axis=0)
    q3 = jnp.quantile(x, 0.75, axis=0)
    iqr = q3 - q1

    scaled_x = jnp.where(
      iqr != 0,
      (x - q1) / (iqr + epsilon_default), 
      0.0 
    )
    return scaled_x

  def backward(self, incoming_error, x, *args, **kwargs) -> jnp.ndarray:
    q1 = jnp.quantile(x, 0.25, axis=0)
    q3 = jnp.quantile(x, 0.75, axis=0)
    iqr = q3 - q1

    scaled_x = jnp.where(
      iqr != 0,
      (incoming_error * (iqr + epsilon_default)) + q1, 
      0.0 
    )
    return scaled_x
