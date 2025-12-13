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

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
      - fan_in (int): number of incoming connections
      - fan_out_size (int): number of outgoing connections
    - Returns:
      - jnp.ndarray: weight matrix as specified in 'shape'
  """
  @abstractmethod
  def __call__(self, seed:int, shape:tuple, fan_in:int, fan_out_size:int):
    """
    Main method: generates a weight matrix according to the inputs and a specified shape
    
    Args:
      seed (int): seed for the psudeo-random number generator
      shape (tuple): shape of the weight matrix
      fan_in (int): number of incoming connections
      fan_out_size (int): number of outgoing connections
      
    Returns:
      jnp.ndarray: weight matrix as specified in 'shape'
    """
    pass

##########################################################################################################
#                                            Built-in Contents                                           #
##########################################################################################################

class Glorot_Uniform(Initializer):
  def __call__(self, seed:int, shape:tuple, fan_in:int, fan_out_size:int):
    limit = jnp.sqrt(2 / (fan_in + fan_out_size))
    return random.uniform(random.PRNGKey(seed), shape, minval=-limit, maxval=limit)

class Glorot_Normal(Initializer):
  def __call__(self, seed:int, shape:tuple, fan_in:int, fan_out_size:int):
    std_dev = jnp.sqrt(2 / (fan_in + fan_out_size))
    return std_dev * random.normal(random.PRNGKey(seed), shape)

class Kaiming_Uniform(Initializer):
  def __call__(self, seed:int, shape:tuple, fan_in:int, fan_out_size:int):
    limit = jnp.sqrt(12 / fan_in)
    return random.uniform(random.PRNGKey(seed), shape, minval=-limit, maxval=limit)

class Kaiming_Normal(Initializer):
  def __call__(self, seed:int, shape:tuple, fan_in:int, fan_out_size:int):
    std_dev = jnp.sqrt(2 / fan_in)
    return std_dev * random.normal(random.PRNGKey(seed), shape)

class Lecun_Uniform(Initializer):
  def __call__(self, seed:int, shape:tuple, fan_in:int, fan_out_size:int):
    limit = jnp.sqrt(3 / fan_in)
    return random.uniform(random.PRNGKey(seed), shape, minval=-limit, maxval=limit)

class Lecun_Normal(Initializer):
  def __call__(self, seed:int, shape:tuple, fan_in:int, fan_out_size:int):
    std_dev = jnp.sqrt(1 / fan_in)
    return std_dev * random.normal(random.PRNGKey(seed), shape)

class Default(Initializer):
  def __call__(self, seed:int, shape:tuple, fan_in:int, fan_out_size:int):
    return random.uniform(random.PRNGKey(seed), shape, minval=-1, maxval=1)

