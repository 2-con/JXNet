import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax.numpy as jnp

def SinusoidalEmbedding(raw_input):
  
  vectors, dims = raw_input.shape
  
  new_dims = dims + 1 if dims % 2 != 0 else dims
  
  position = jnp.arange(vectors)[:, jnp.newaxis]
  div_term = jnp.exp(jnp.arange(0, new_dims, 2) * -(jnp.log(10000.0) / new_dims))
  
  pe = jnp.zeros((vectors, new_dims))
  pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
  pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

  return raw_input + pe if dims % 2 == 0 else raw_input + pe[:, :-1]

def OneHot(data):
  if isinstance(data, list) and all(isinstance(item, str) for item in data):
    unique_classes = sorted(list(set(data)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    indices = jnp.array([class_to_idx[cls] for cls in data])
    num_classes = len(unique_classes)
  
  elif isinstance(data, jnp.ndarray) and jnp.issubdtype(data.dtype, jnp.integer):
    unique_classes = jnp.unique(data)
    num_classes = len(unique_classes)
    indices = data
    
  else:
    raise TypeError("Input must be a Python list of strings or a JAX array of integers.")
  
  one_hot_matrix = jnp.zeros((len(data), num_classes), dtype=jnp.float32)
  one_hot_matrix = one_hot_matrix.at[jnp.arange(len(data)), indices].set(1)
  
  return one_hot_matrix

def OrdinalEncoder(ranking, data):

  ranks = {value: index for index, value in enumerate(ranking)}
  
  def _raise_missing(element):
    raise ValueError(f"'{element}' not found in ranking list.")
  
  if isinstance(data, (list, tuple)):
    encoded = [ranks[element] if element in ranks else _raise_missing(element) for element in data]
    return jnp.array(encoded, dtype=jnp.int32)

  elif isinstance(data, jnp.ndarray):
    def encode_elem(x):
      val = x.item()
      if val not in ranks:
        raise ValueError(f"'{val}' is not found in ranking list.")
      return ranks[val]
    return jnp.array([encode_elem(x) for x in data], dtype=jnp.int32)

  else:
    raise TypeError("Input must be a list, tuple, or JAX array.")

