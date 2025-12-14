"""
Initializers
=====
  Initializers are used to generate weight matrices according to a specified shape and initialization scheme to stabilize gradient flow

Provides:
- Initializer base class
  - The base class all JXNet initializers must inherit and follow.
    Contains scaffolding for custom and built-in layers.

- Default (Uniform [-1, 1])
- Glorot Uniform
- Glorot Normal
- Kaiming Normal (He Normal)
- Kaiming Uniform (He Uniform)
- LeCun Normal
- LeCun Uniform
"""

from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import random

class Initializer(ABC):
  """
  Base class for all initializers
  
  An initializer class is required to have the following:
  - '__call__' : A method that generates a weight matrix according to the inputs and a specified shape
    - Args:
      - shape (tuple): shape of the weight matrix
      - fanin_shape (int): number of incoming connections
      - fanout_shape (int): number of outgoing connections
    - Returns:
      - jnp.ndarray: weight matrix as specified in 'shape'
  """
  @abstractmethod
  def __call__(self, seed:int, shape:tuple, fanin_shape:tuple, fanout_shape:tuple):
    """
    Main method: generates a weight matrix according to the inputs and a specified shape
    
    Args:
      seed (int): seed for the psudeo-random number generator
      shape (tuple): shape of the weight matrix
      fanin_shape (int): number of incoming connections
      fanout_shape (int): number of outgoing connections
      
    Returns:
      jnp.ndarray: weight matrix as specified in 'shape'
    """
    pass

##########################################################################################################
#                                            Built-in Contents                                           #
##########################################################################################################

class Glorot_Uniform(Initializer):
  @staticmethod
  def __call__(seed:int, shape:tuple, fan_in:tuple, fan_out_size:tuple):
    fan_in_scalar = jnp.prod(jnp.array(fan_in))
    fan_out_scalar = jnp.prod(jnp.array(fan_out_size))
    limit = jnp.sqrt(6.0 / (fan_in_scalar + fan_out_scalar)) # Use 6.0 for uniform
    return random.uniform(random.PRNGKey(seed), shape, minval=-limit, maxval=limit)

class Glorot_Normal(Initializer):
  @staticmethod
  def __call__(seed:int, shape:tuple, fan_in:tuple, fan_out_size:tuple):
    fan_in_scalar = jnp.prod(jnp.array(fan_in))
    fan_out_scalar = jnp.prod(jnp.array(fan_out_size))
    std_dev = jnp.sqrt(2.0 / (fan_in_scalar + fan_out_scalar))
    return std_dev * random.normal(random.PRNGKey(seed), shape)

class Kaiming_Uniform(Initializer):
  @staticmethod
  def __call__(seed:int, shape:tuple, fan_in:tuple, fan_out_size:tuple):
    fan_in_scalar = jnp.prod(jnp.array(fan_in))
    limit = jnp.sqrt(6.0 / fan_in_scalar) # Use 6.0 for uniform (equivalent to sqrt(12/2/fan_in))
    return random.uniform(random.PRNGKey(seed), shape, minval=-limit, maxval=limit)

class Kaiming_Normal(Initializer):
  @staticmethod
  def __call__(seed:int, shape:tuple, fan_in:tuple, fan_out_size:tuple):
    fan_in_scalar = jnp.prod(jnp.array(fan_in))
    std_dev = jnp.sqrt(2.0 / fan_in_scalar)
    return std_dev * random.normal(random.PRNGKey(seed), shape)

class Lecun_Uniform(Initializer):
  @staticmethod
  def __call__(seed:int, shape:tuple, fan_in:tuple, fan_out_size:tuple):
    fan_in_scalar = jnp.prod(jnp.array(fan_in))
    limit = jnp.sqrt(3.0 / fan_in_scalar)
    return random.uniform(random.PRNGKey(seed), shape, minval=-limit, maxval=limit)

class Lecun_Normal(Initializer):
  @staticmethod
  def __call__(seed:int, shape:tuple, fan_in:tuple, fan_out_size:tuple):
    fan_in_scalar = jnp.prod(jnp.array(fan_in))
    std_dev = jnp.sqrt(1.0 / fan_in_scalar)
    return std_dev * random.normal(random.PRNGKey(seed), shape)

class Default(Initializer):
  @staticmethod
  def __call__(seed:int, shape:tuple, fanin_shape:tuple, fanout_shape:tuple):
    return random.uniform(random.PRNGKey(seed), shape, minval=-1, maxval=1)

class Ones(Initializer):
  """
  Ones
  -----
    Initializes all weights to one. This is mainly for testing purposes and is not recommended for actual model training.
  """
  @staticmethod
  def __call__(seed:int, shape:tuple, fanin_shape:tuple, fanout_shape:tuple):
    return jnp.ones(shape)

