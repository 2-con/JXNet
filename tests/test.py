import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models.regressor as Regressors
from core.standard import optimizers, losses
import jax.numpy as jnp

dymmy = jnp.array([
  [0],
  [1],
  [2],
  [3],
  [4],
  [5]
])

model = Regressors.Power(True,1,1)
model.compile(
  losses.Mean_Squared_Error(),
  optimizers.Default(),
  0.01,
  10,
  verbose=2,
  logging=1,
)
model.fit(dymmy, dymmy)
print(model.params)