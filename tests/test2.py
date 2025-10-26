import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.standardnet import Sequential
from core.standard import layers, optimizers, losses
import time
import jax

# Example Usage
model = Sequential()
model.add(layers.Dense(2, "Identity"))
# model.add(layers.Dense(2, "leaky relu"))

# Compile the model
model.compile(
  input_shape=(2,),
  loss=losses.Mean_Squared_Error(),
  optimizer=optimizers.RMSprop(),
  learning_rate=0.01,
  epochs=10,
  verbose=2,
  logging=1,
)

# some dummy data for training
features = jax.numpy.array([[0,0],[0,1],[1,0],[1,1]])
targets = jax.numpy.array([[0,1],[1,0],[1,0],[0,1]])

# jnp.array([[0,0],[0,1],[1,0],[1,1]])
# jnp.array([[0,1],[1,0],[1,0],[0,1]])

# Fit the model
start = time.perf_counter()

model.fit(features, targets) 

print(f"""
      Finished training in {time.perf_counter() - start} seconds
      """)