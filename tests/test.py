import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import api.netlab as nl
import api.standardnet as sn
from core.lab.procedures import *
from core.standard.layers import Dense
from core.standard.functions import ReLU
from core.standard.optimizers import Default
from core.standard.losses import Mean_Squared_Error
import jax.numpy as jnp

model = nl.Sample(
  sn.Sequential(
    Dense(3, ReLU()),
    Dense(2, ReLU())
  )
)

model.procedure(
  Compile(
    input_shape=(2,),
    optimizer = Default(learning_rate=0.01),
    loss = Mean_Squared_Error(),
    epochs = 2,
    verbose = 0
  ),
  Train("features", "targets"),
  Track_Layer(
    datapath = ("layer_0", "weights")
  )
)

model.compile(
  cycles = 2,
  verbose = 2,
  logging = 1,
)

model.run({
  "Test": {
    "features": jnp.array([[0,0],[0,1],[1,0],[1,1]]),
    "targets": jnp.array([[0,1],[1,0],[1,0],[0,1]])
  }
})

print(model.logs["Test"]["Model 1"])

