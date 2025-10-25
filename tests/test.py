import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.regressor import Linear
from core.standard import optimizers, losses
import jax.numpy as jnp

dymmy = jnp.array(
  [
    [1],
    [2],
    [3]
  ]
)

model = Linear(1,1)
model.compile(
  losses.Mean_Squared_Error(),
  optimizers.Adam(),
  0.001,
  500,
  verbose=2
)
model.fit(dymmy, dymmy)
print(model.predict( jnp.array([1]) ))