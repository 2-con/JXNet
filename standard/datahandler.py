import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax
import jax.numpy as jnp
import random

def batch_data(batchsize:int, features:jnp.ndarray, targets:jnp.ndarray):
  """
  Batch Data
  -----
    Batch data into batches of size batchsize.
  -----
  Args
  -----
  - batchsize (int)         : The size of each batch
  - features  (jnp.ndarray) : The features to be batched
  - targets   (jnp.ndarray) : The targets to be batched
  -----
  Returns
  -----
  batched_features (jnp.ndarray) : The batched features
  batched_targets  (jnp.ndarray) : The batched targets
  """
  key = jax.random.PRNGKey(random.randint(0, 2**32))
  shuffled_indices = jax.random.permutation(key, features.shape[0])

  # Determine the number of complete batches
  num_batches = len(features) // batchsize
  shuffled_features = features[shuffled_indices][:num_batches * batchsize]
  shuffled_targets = targets[shuffled_indices][:num_batches * batchsize]

  # Reshape data into (num_batches, batch_size, ...)
  batched_features = shuffled_features.reshape(num_batches, batchsize, *features.shape[1:])
  batched_targets = shuffled_targets.reshape(num_batches, batchsize, *targets.shape[1:])

  return batched_features, batched_targets

def split_data(features:jnp.ndarray, targets:jnp.ndarray, split:float):
  """
  Split Data
  -----
    Splits data into two sets of configurable lengths, used for splitting data into training and testing sets
  -----
  Args
  -----
  - features  (jnp.ndarray) : The features to be split
  - targets   (jnp.ndarray) : The targets to be split
  - split     (float)       : The fraction of the data to be used for the first set
  -----
  Returns
  -----
    (1st set features, 1st set targets)
  """
  
  if not 0 <= split <= 1:
    raise ValueError("Split must be between 0 and 1")
  
  return jnp.asarray(features[0:int(len(features)*split)]), jnp.asarray(targets[0:int(len(targets)*split)])

