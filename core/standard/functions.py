"""
Functions
=====
  Activation functions are functions that transform the output of a layer into another value to introduce non-linearity. Scalers are functions that transform 
  the output of a layer into another value to scale them to stablize the outputs.
  
  All functions within this file is guarenteed to inherit from a strict Function base class which ensures the existance of a forward pass method
  and a backward pass (derivative) method which automatically returns the gradient given an input value and the incoming error.

Provides:
- Function base class
  - The base class all JXNet functions must inherit and follow.
    Contains scaffolding for custom and built-in layers.

- Sigmoid
  - Sigmoid can be converted to a training-only Identity function through the constructor. During inference, it becomes the proper sigmoid function.
  - Sigmoid's aliasiing (properly called the 'return_logits' parameter) is to allow the Binary Cross Entropy loss function to increase numerical stability and allow for stable backpropagation.
    Though, this behavior is set to False by default.
- Tanh
- Binary Step (Heavyside Step Function) where y = 0 when x = 0
- Softsign
- Softmax
  - Softmax is an alias for the Identity function during training. Only during inference will it become the proper softmax function.
  - Softmax is an alias to allow the Cross-Entropy loss function to simplify the jacobian and allow for accurate backpropagation.
- ReLU (Rectified Linear Unit)
- Softplus
- Mish
- Swish
- Leaky ReLU (Leaky Rectified Linear Unit)
- GELU (Gaussian Error Linear Unit)
- Identity
- ELU (Exponential Linear Unit)
- SELU (Scaled Exponential Linear Unit)
- PReLU (Parametric Rectified Linear Unit)
- Swish-Beta

- Standard Scaler
- Min-Max Scaler
- Max-Abs Scaler
- Robust Scaler
"""
import jax.numpy as jnp
from abc import ABC, abstractmethod
import jax

class Function(ABC):
  """
  Base class for all functions compatible with JXNet layers. 
  -----
    Function classes are used to apply a mathematical function to an array and is only used inside a Layer class.
  
  ### A Function class is required to have the following:
  - 'forward' : method for applying the Function function.
    - Args:
      - x (jnp.ndarray): The input array to the Function function.
      - *args: Variable length argument list.  Can be used to pass additional information to the Function function.
      - **kwargs: Arbitrary keyword arguments. Used to pass parameters (if any) to the Function function. 
                Make sure the parameter names match those listed in the 'parameters' attribute.
    - Returns:
      - jnp.ndarray: The output array after applying the Function function, with the same dimensions as the input.
  
  - 'backward' : method for computing the gradient of the Function function.
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

##########################################################################################################
#                                            Built-in Contents                                           #
##########################################################################################################

# normalization
class Sigmoid(Function):
  """
  Sigmoid
  -----
    Sigmoid activation function. While Binary Cross-Entropy will simplify the backwards error, this implimentation will not simplify the backwards error in order to be flexible.
    Unlike the softmax function, the sigmoid function have a simple derivative.
  
  args:
  - return_logits (bool): If True, the function will return the logits instead of the activations during training only.
  
  ### Math
    sigmoid(x) = 1 / (1 + exp(-x))
  """
  def __init__(self, return_logits=False):
    self.return_logits = return_logits
  
  def forward(self, x, *args, **kwargs):
    if self.return_logits and kwargs.get("training"):
      return x
    return jax.nn.sigmoid(x)

  def backward(self, incoming_error, x, *args, **kwargs):
    if self.return_logits and kwargs.get("training"): # If training, return the gradient of the loss with respect to the logits
      return {"x": incoming_error}
    
    local_grad = (1.0 / (1.0 + jnp.exp(-x))) * (1 - (1.0 / (1.0 + jnp.exp(-x))))
    return {"x": incoming_error * local_grad} # Outputs dL/dz
  
class Tanh(Function):
  """
  Tanh
  -----
    Tanh activation function.
  
  ### Math
    tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
  """
  @staticmethod
  def forward(x, *args, **kwargs):
    return jnp.tanh(x)

  @staticmethod
  def backward(self, incoming_error, x, *args, **kwargs):
    local_grad = 1 - jnp.tanh(x)**2
    return {"x": incoming_error * local_grad} # Outputs dL/dz

class Binary_step(Function):
  """
  Binary Step
  -----
    Binary Step activation function, also referred to as the Heaviside step function.
  
  ### Math
    heaviside(x) = 1 if x > 0 else 0
  """
  @staticmethod
  def forward(x, *args, **kwargs):
    return jnp.where(x > 0, 1.0, 0.0)
  
  @staticmethod
  def backward(incoming_error, x, *args, **kwargs):
    return {"x": jnp.zeros_like(x)} # dL/dz is 0

class Softsign(Function):
  """
  Softsign
  -----
    Softsign activation function.
  
  ### Math
    softsign(x) = x / (1 + |x|)
  """
  @staticmethod
  def forward(x, *args, **kwargs):
    return x / (1.0 + jnp.abs(x))
  
  @staticmethod
  def backward(self, incoming_error, x, *args, **kwargs):
    local_grad = 1 / (1 + jnp.abs(x))**2
    return {"x": incoming_error * local_grad} # Outputs dL/dz

class Softmax(Function):
  """
  Softmax
  -----
    While softmax is differentiable, the output is a jacobian for any arbritrary input without any external simplifications.
    therefore, this softmax class is more of a control function rather than a proper function with 100% accurate derivative for
    backpropagation.
  
  ### During Training
    The softmax function is an alias for the identity function. This is the default function to substitute since it preserves activations
    during training, which is essencial when cross-entropy loss applies the proper softmax to simplify the jacobian.
  
  ### During Inference
    The softmax function is a proper softmax function. This is because the backward gradient is not computed during Inference.
  """
  def forward(self, x, *args, **kwargs):
    if kwargs.get("training"):
      return x
    else:
      exp_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
      return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

  def backward(self, incoming_error, x, *args, **kwargs):
    return {"x": incoming_error}

class ReLU(Function):
  """
  ReLU (Rectified Linear Unit)
  -----
    ReLU activation function.
  
  ### Math
    relu(x) = max(0, x)
  """
  @staticmethod
  def forward(x, *args, **kwargs):
    return jnp.maximum(0.0, x)

  @staticmethod
  def backward(incoming_error, x, *args, **kwargs):
    local_grad = jnp.where(x > 0, 1.0, 0.0)
    return {"x": incoming_error * local_grad} # Outputs dL/dz

class Softplus(Function):
  """
  Softplus
  -----
    Softplus activation function. A smooth approximation to ReLU.
  
  ### Math
    softplus(x) = log(1 + exp(x))
  """
  @staticmethod
  def forward(x, *args, **kwargs):
    return jnp.log(1.0 + jnp.exp(x))

  @staticmethod
  def backward(incoming_error, x, *args, **kwargs):
    local_grad = 1 / (1 + jnp.exp(-x))
    return {"x": incoming_error * local_grad} # Outputs dL/dz

class Mish(Function):
  """
  Mish
  -----
    Mish activation function.
  
  ### Math
    mish(x) = x * tanh(log(1 + exp(x)))
  """
  @staticmethod 
  def forward(x, *args, **kwargs):
    return x * jnp.tanh(jnp.log(1.0 + jnp.exp(x)))
  
  @staticmethod
  def backward(incoming_error, x, *args, **kwargs):
    local_grad = jnp.tanh(jnp.log(1.0 + jnp.exp(x))) + (x * (jnp.exp(x) / (jnp.exp(x) + 1.0)) * (1 - jnp.tanh(jnp.log(1.0 + jnp.exp(x)))**2))
    return {"x": incoming_error * local_grad}

class Swish(Function):
  """
  Swish
  -----
    Swish activation function, also referred to as SiLU (Sigmoid Linear Unit).
  
  ### Math
    swish(x) = x * sigmoid(x)
  """
  @staticmethod
  def forward(x, *args, **kwargs):
    return x * (1.0 / (1.0 + jnp.exp(-x)))

  @staticmethod
  def backward(incoming_error, x, *args, **kwargs):
    s = (1.0 / (1.0 + jnp.exp(-x))) # Sigmoid(x)
    s_prime = s * (1 - s)           # Sigmoid'(x)
    local_grad = s + x * s_prime    # d(x*s)/dx
    return {"x": incoming_error * local_grad}

class Leaky_ReLU(Function):
  """
  Leaky ReLU
  -----
    Leaky ReLU activation function.
  
  ### Math
    leaky_relu(x) = max(0.1 * x, x)
  """
  @staticmethod
  def forward(x, *args, **kwargs):
    return jnp.maximum(0.1 * x, x)

  @staticmethod
  def backward(incoming_error, x, *args, **kwargs):
    local_grad = jnp.where(x > 0, 1.0, 0.1)
    return {"x": incoming_error * local_grad}

class GELU(Function):
  """
  GELU (Gaussian Error Linear Unit)
  -----
    GELU activation function. 
  
  ### Math
    gelu(x) = x * NormalCDF(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
  """
  @staticmethod
  def forward(x, *args, **kwargs):
    return jax.nn.gelu(x)

  @staticmethod
  def backward(incoming_error, x, *args, **kwargs):
    local_grad = (jax.nn.gelu(x)/x) + x*jnp.exp(-x**2)
    return {"x": incoming_error * local_grad}

class Identity(Function): 
  """
  Identity
  -----
    Identity function, the default activation function.
  
  ### Math
    Identity(x) = x
  """
  @staticmethod 
  def forward(x, *args, **kwargs):
    return x

  @staticmethod
  def backward(incoming_error, x, *args, **kwargs):
    return {"x": incoming_error}

class ELU(Function):
  """
  ELU (Exponential Linear Unit)
  -----
    ELU activation function.
  
  ### Math
    elu(x) = x if x > 0 else exp(x) - 1
  """
  @staticmethod
  def forward(x, *args, **kwargs):
    return jnp.where(x > 0, x, (jnp.exp(x) - 1.0))

  @staticmethod
  def backward(incoming_error, x, *args, **kwargs):
    local_grad_x = jnp.where(x > 0, 1.0, jnp.exp(x))
    
    return {
      "x": incoming_error * local_grad_x
    }

class SELU(Function):
  """
  SELU (Scaled Exponential Linear Unit)
  -----
    SELU activation function with approximate default constant parameters.
  
  ### Math
    SELU(x, alpha, beta) = beta * (x if x > 0 else alpha * (exp(x) - 1))
  """
  def __init__(self, alpha=1.6732, beta=1.0507, *args, **kwargs):
    self.alpha = alpha
    self.beta = beta
  
  def forward(self, x, *args, **kwargs):
    return self.beta * jnp.where(x > 0, x, self.alpha * (jnp.exp(x) - 1.0))

  def backward(self, incoming_error, x, *args, **kwargs):
    local_grad_x = self.beta * jnp.where(x > 0, 1.0, self.alpha * jnp.exp(x))
    local_grad_alpha = self.beta * jnp.where(x <= 0, (jnp.exp(x) - 1.0), 0.0)
    
    return {
      "x": incoming_error * local_grad_x,
      "alpha": jnp.sum(incoming_error * local_grad_alpha),
      "beta": jnp.sum(incoming_error * jnp.where(x > 0, x, (self.alpha * jnp.exp(x) - 1.0)))
    }
    
########################################################################################################################
#                                           parametric Functions                                                       #
########################################################################################################################

class PReLU(Function):
  """
  PReLU (Parametric Rectifier Linear Unit)
  -----
    PReLU activation function. Alpha is a parameter that is optimized per-layer.
  
  ### Math
    PReLU(x, alpha) = max(alpha * x, x)
  """
  parameters = ["alpha"]
  
  @staticmethod
  def forward(x, alpha, *args, **kwargs):
    return jnp.maximum(alpha * x, x)

  @staticmethod
  def backward(incoming_error, x, alpha, *args, **kwargs):
    local_grad_x = jnp.where(x > 0, 1.0, alpha)
    local_grad_alpha = jnp.where(x <= 0, x, 0.0)
    
    return {
      "x": incoming_error * local_grad_x,
      "alpha": jnp.sum(incoming_error * local_grad_alpha)
    }

class Swish_beta(Function):
  """
  Swish-Beta
  -----
    Swish-Beta activation function. Alpha is a parameter that is optimized per-layer.
  
  ### Math
    Swish-Beta(x, beta) = x / (1 + exp(-x * beta))
  """
  parameters = ["alpha"]
  
  @staticmethod
  def forward(x, alpha, *args, **kwargs):
    return x * (1.0 / (1.0 + jnp.exp(-alpha * x)))

  @staticmethod
  def backward(incoming_error, x, alpha, *args, **kwargs):
    s = (1.0 / (1.0 + jnp.exp(-alpha * x))) # Sigmoid(alpha * x)
    s_prime = s * (1 - s)                   # Sigmoid'(alpha * x)
    
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
  """
  Standard Scaler
  -----
    Standard Scaler, scales the distribution to a mean of 0 and a variance of 1. It is essensially layernorm without running mean and variance.
  """
  @staticmethod
  def forward(x:jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
    average = jnp.mean(x, axis=0)
    standard_deviation = jnp.std(x, axis=0)

    return jnp.where(
      standard_deviation != 0,
      (x - average) / (standard_deviation + 1e-8),
      0.0
    )

  @staticmethod
  def backward(incoming_error, x, *args, **kwargs) -> jnp.ndarray:
    average = jnp.mean(x, axis=0)
    standard_deviation = jnp.std(x, axis=0)

    scaled_x = jnp.where(
      standard_deviation != 0,
      (incoming_error * (standard_deviation + 1e-8)) + average,
      0.0
    )
    return scaled_x

class Min_Max_Scaler(Function):
  """
  Min-Max Scaler
  -----
    Minimum-Maximum Scaler, normalizes a distribution to range between 0 and 1 inclusive [0,1].
  """
  @staticmethod
  def forward(x:jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
    max_val = jnp.max(x, axis=0)
    min_val = jnp.min(x, axis=0)
    range_val = max_val - min_val
    
    scaled_x = jnp.where(
      range_val != 0,
      (x - min_val) / (range_val + 1e-8),
      0.0 
    )
    return scaled_x

  @staticmethod
  def backward(incoming_error, x, *args, **kwargs) -> jnp.ndarray:
    min_val = jnp.min(x, axis=0)
    max_val = jnp.max(x, axis=0)
    range_val = max_val - min_val
    
    scaled_x = jnp.where(
      range_val != 0,
      (incoming_error * (max_val + 1e-8)) + min_val,
      0.0 
    )
    return scaled_x

class Max_Abs_Scaler(Function):
  """
  Max-Abs Scaler
  -----
    Maximum-Abseloute Value Scaler, a spesific case of the Min-Max Scaler where all values in a distribution is normalized such that the values are
    between 0 and 1 inclusive [0,1] if all values are positive, between -1 and 0 if all values are negative, and between -1 and 1 if there are both signs present.
    """
  @staticmethod
  def forward(x:jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
    max_abs_val = jnp.max(jnp.abs(x), axis=0)

    scaled_x = jnp.where(
      max_abs_val != 0,
      x / (max_abs_val + 1e-8),
      0.0
    )
    return scaled_x

  @staticmethod
  def backward(incoming_error, x, *args, **kwargs) -> jnp.ndarray:
    max_abs_val = jnp.max(jnp.abs(x), axis=0)

    scaled_x = jnp.where(
      max_abs_val != 0,
      incoming_error * (max_abs_val + 1e-8),
      0.0
    )
    return scaled_x

class Robust_Scaler(Function):
  """
  Robust Scaler
  -----
    Robust Scaler, scales data points according to the mean and interqartile range instead of the full data.
  """
  @staticmethod
  def forward(x:jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
    q1 = jnp.quantile(x, 0.25, axis=0)
    q3 = jnp.quantile(x, 0.75, axis=0)
    iqr = q3 - q1

    scaled_x = jnp.where(
      iqr != 0,
      (x - q1) / (iqr + 1e-8), 
      0.0 
    )
    return scaled_x

  @staticmethod
  def backward(incoming_error, x, *args, **kwargs) -> jnp.ndarray:
    q1 = jnp.quantile(x, 0.25, axis=0)
    q3 = jnp.quantile(x, 0.75, axis=0)
    iqr = q3 - q1

    scaled_x = jnp.where(
      iqr != 0,
      (incoming_error * (iqr + 1e-8)) + q1, 
      0.0 
    )
    return scaled_x
