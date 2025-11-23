import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.standardnet import Sequential
from core.standard import layers, optimizers, losses, functions
import time
import jax

# Example Usage
model = Sequential(
  layers.Recurrent(2, functions.ReLU()),
  layers.Recurrent(2, functions.ReLU(), output_sequence=(1,)),
)

# Compile the model
model.compile(
  input_shape=(2,2),
  loss=losses.Mean_Squared_Error(),
  optimizer=optimizers.Default(0.001),
  epochs=100,
  verbose=2,
  logging=1,
)

# some dummy data for training
features = jax.numpy.array([
  [[0,1],
   [1,0]],
  
  [[1,0],
   [0,1]],
  
  [[1,1],
   [0,0]],
  
  [[0,0],
   [1,1]]
], dtype=jax.numpy.float32)

targets = jax.numpy.array([
  [[0,1]],
  [[0,1]],
  [[1,0]],
  [[1,0]]
], dtype=jax.numpy.float32)

# Fit the model
start = time.perf_counter()

model.fit(features, targets)

print(model.push(features))

print(f"""
      Finished training in {time.perf_counter() - start} seconds
      """)