"""
StandardNet API
=====
  A mid-level API for sequential models of networks. Unlike StaticNet, StandardNet is intended to be built for speed and efficiency
  while staying modular by using JAX and other libraries.
"""
#######################################################################################################
#                                    File Information and Handling                                    #
#######################################################################################################

__version__ = "1.0.0"

__package__ = "pynet"

if __name__ == "__main__":
  print("""
        This file is not meant to be run as a main file.
        More information can be found about PyNet's StandardNet API on the documentation.
        system > 'docs.txt' or the GitHub repository at https://github.com/2-con/PyNet
        """)
  exit()

#######################################################################################################
#                                               Imports                                               #
#######################################################################################################

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import random

from tools.utility import progress_bar
from core.standard import metrics, optimizers, losses, initializers, functions, callbacks, datahandler
from system.config import *

#######################################################################################################
#                                               Extra                                                 #
#######################################################################################################

""" Notes:

jax.grad and jax.value_and_grad dosn't work for some reason, it keeps breaking for no reason at all.
chatgpt says its a new bug and that i should be reported. whatever the cause, those are off the table.

yes, i've tried to fix it for hours and it still doesn't work.
"""

#######################################################################################################
#                                          Sequential Model                                           #
#######################################################################################################

class Sequential:
  __api__ = "StandardNet"
  # pre-processing

  def __init__(self, *args):
    """
    Sequential
    ======
      Sequential model where layers are processed sequentially.

      Must contain Standard or Standard-inherited layers to be added to the model. High-level string arguments will not be 
      translated into their respective instances, for that use the .add() method.
    """
    self.layers = list(args) if args is not None else []

    self.error_logs = []
    self.validation_error_logs = []
    self.metric_logs = []

  def add(self, layer):
    """
    Add
    -----
      Translates string arguments into their respective instances and adds the layer to the model.
    -----
    Args
    -----
    - layer (StandardNet Instance) : the layer to add to the model **without** parameters
    -----
    Examples
    -----
      >>> 
    model = Sequential(
      Dense(64, ReLU()),
      Dense(32, ReLU()),
      Dense(16, ReLU()),
    )
          
      Can be written as
      
      >>> 
      model = Sequential()
      model.add(Dense(64, "ReLU"),
      model.add(Dense(32, "ReLU"),
      model.add(Dense(16, "ReLU"),
        
      Note that originally integer parameters are also passed as an integer in this new format. String are not case-sensitive
      though it is reccomended to correctly capitalize for readability.
    """
  
    class Key:
      FUNCTION = {
        # normalization activations
        "sigmoid": functions.Sigmoid(),
        "tanh": functions.Tanh(),
        "binary step": functions.Binary_step(),
        "softsign": functions.Softsign(),
        "softmax": functions.Softmax(),
        
        # rectifiers
        "relu": functions.ReLU(),
        "softplus": functions.Softplus(),
        "mish": functions.Mish(),
        "swish": functions.Swish(),
        "leaky relu": functions.Leaky_ReLU(),
        "gelu": functions.GELU(),
        "identity": functions.Linear(),
        "reeu": functions.ReEU(),
        "retanh": functions.ReTanh(),
        
        # parametric activations
        'elu': functions.ELU(),
        "selu": functions.SELU(),
        "prelu": functions.PReLU(),
        "silu": functions.SiLU(),
        
        "standard scaler": functions.Standard_Scaler(),
        "min max scaler": functions.Min_Max_Scaler(),
        "max abs scaler": functions.Max_Abs_Scaler(),
        "robust scaler": functions.Robust_Scaler(),
      }
      
      INITIALIZER = {
        "glorot uniform": initializers.Glorot_Uniform,
        "glorot normal": initializers.Glorot_Normal,
        "he uniform": initializers.He_Uniform,
        "he normal": initializers.He_Normal,
        "lecun uniform": initializers.Lecun_Uniform,
        "lecun normal": initializers.Lecun_Normal,
        "xavier uniform in": initializers.Xavier_Uniform_In,
        "xavier uniform out": initializers.Xavier_Uniform_Out,
        "default": initializers.Default
      }

    if hasattr(layer, "initializer"):
      layer.initializer = Key.INITIALIZER['default']()
      if layer.function.lower() in ('relu','softplus','mish','swish','leaky relu','gelu','reeu','none','retanh','elu', 'selu', 'prelu', 'silu'):
        layer.initializer = Key.INITIALIZER['he normal']()
      elif layer.function.lower() in ('binary step','softsign','sigmoid','tanh','softmax'):
        layer.initializer = Key.INITIALIZER['glorot normal']()
    
    if hasattr(layer, "function"):
      if layer.function.lower() in Key.FUNCTION:
        layer.function = Key.FUNCTION[layer.function.lower()]
      else:
        raise Exception(f"Unknown activation function: {layer.function}. Available: {list(Key.FUNCTION.keys())}")
    
    self.layers.append(layer)

  def compile(self, input_shape:tuple[int,...], optimizer:optimizers.Optimizer, loss:losses.Loss, learning_rate:float, epochs:int, metrics:tuple[metrics.Metric,...]=[], validation_split:float=0, batch_size:int=1, verbose:int=1, logging:int=1, optimizer_hyperparameters:dict={}, *args, **kwargs):
    """
    Compile
    -----
      Compiles the model to be ready for training, but unlike NetCore, the input shape needs to be defined here
      so the model can be compiled faster. Custom callables can be passed as a parameter as long as they are compatible with JAX (JNP-based functions).
    -----
    Args
    -----
    - input_shape                 (tuple[int, ...])                    : shape of the input data, include channels for image data and features for tabular data.
    - loss                        (core.standard.losses.Loss)          : loss function to use, not an instance
    - optimizer                   (core.standard.optimizers.Optimizer) : optimizer to use, not an instance
    - learning_rate               (float)                              : learning rate to use
    - epochs                      (int)                                : number of epochs to train for
    
    - (Optional) metrics           (list)                               : metrics to evaluate. can be a list of core.standard.metrics.Metric or a core.standard.losses.Loss instance
    - (Optional) batch_size        (int)                                : batch size to use
    - (Optional) verbose           (int)                                : verbosity level
    - (Optional) logging           (int)                                : how ofter to report if the verbosity is at least 3
    - (Optional) callbacks         (core.standard.callback)             : call a custom callback class during training with access to all local variables, read more in the documentation.
    - (Optional) validation_split  (float)                              : fraction of the data to use for validation, must be between [0, 1). Default is 0 (no validation).
    
    - (Optional) regularization   (tuple[str, float])                  : type of regularization to use, position 0 is the type ("L1" or "L2"), position 1 is the lambda value. Default is None (no regularization).
    - (Optional) universal_seed   (int)                                : seed to use for all random operations, default is None (randomly generated).
    
    Verbosity Levels
    -----
    - 0 : None
    - 1 : Progress bar of the whole training process
    - 2 : (Numerical output) Loss
    - 3 : (Numerical output) Loss and V Loss (Validation Loss)
    - 4 : (Numerical output) Loss, V Loss (Validation Loss) and the 1st metric in the 'metrics' list
    """
    self.input_shape = input_shape
    self.learning_rate = jnp.float32(learning_rate)
    self.epochs = epochs
    self.batchsize = batch_size
    self.verbose = verbose
    self.logging = logging
    
    self.callback = kwargs.get('callback', callbacks.Callback)
    self.optimizer = optimizer
    self.loss = loss
    self.validation_split = validation_split
    self.metrics = tuple([m for m in metrics])
    self.regularization = kwargs.get('regularization', ["None", 1])
    self.universal_seed = kwargs.get('universal_seed', None)
    
    self.error_logs = []
    self.validation_error_logs = []
    self.metrics_logs = []
    self.is_compiled = True
    
    ############################################################################################
    #                                General Error Prevention                                  #
    ############################################################################################
    
    if verbose < 0 or verbose > 4:
      raise ValueError("Verbosity level must be between 0 and 4.")
    if type(verbose) != int:
      raise TypeError("Verbosity level must be an integer.")
    if type(input_shape) != tuple:
      raise TypeError("Input shape must be a tuple.")
    if type(metrics) != list:
      raise TypeError("Metrics must be a list.")
    if type(learning_rate) != float:
      raise TypeError("Learning rate must be a float.")
    if type(epochs) != int:
      raise TypeError("Epochs must be an integer.")
    if not (0 <= validation_split < 1):
      raise ValueError("validation_split must be between [0,1)")
    if validation_split == 0 and len(self.metrics) > 0:
      raise ValueError("Validation split must be > 0 if metrics are used")
    
    ############################################################################################
    #                        Initialize model parameters and connections                       #
    ############################################################################################
    
    self.params_pytree = {} # Initialize the main parameters dictionary here
    sizes = []
    
    # set sizes while ignoring fans
    for layer_index, layer in enumerate(self.layers):
      layer.universal_seed = random.randint(0, 2**32) if self.universal_seed is None else self.universal_seed
      
      if layer_index == 0:
        _, layer_size = layer.calibrate(fan_in_shape=input_shape, fan_out_shape=(1,))
      else:
        _, layer_size = layer.calibrate(fan_in_shape=sizes[layer_index-1], fan_out_shape=(1,))
      
      sizes.append(layer_size)
    
    for layer_index, layer in enumerate(self.layers):
      # Pass the individual key to the layer's init_params method
      
      if len(sizes) == 1:
        layer_params, _ = layer.calibrate(fan_in_shape=input_shape, fan_out_shape=sizes[layer_index])
      
      elif layer_index == 0: # If this is the first layer, use the input shape
        layer_params, _ = layer.calibrate(fan_in_shape=input_shape, fan_out_shape=sizes[layer_index+1])
      
      elif layer_index == len(self.layers) - 1: # If this is the last layer, use the output shape of the previous layer
        layer_params, _ = layer.calibrate(fan_in_shape=sizes[layer_index-1], fan_out_shape=sizes[layer_index])
      
      else:
        layer_params, _ = layer.calibrate(fan_in_shape=sizes[layer_index-1], fan_out_shape=sizes[layer_index+1])
      
      if layer_params: # if layer_params is not empty
        
        if type(layer_params) is not dict:
          raise TypeError(
            f"Layer {layer_index} ({type(layer).__name__})'s init_params method returned parameters of type "
            f"{type(layer_params)}. Expected a dictionary of parameters for learnable layers."
          )
        
        self.params_pytree[f'layer_{layer_index}'] = layer_params # Store layer's params under a unique key

    ############################################################################################
    #                                  Initialize optimizer                                    #
    ############################################################################################
    
    self.opt_state = jax.tree.map(
      lambda p: self.optimizer.initialize(p.shape, p.dtype),
      self.params_pytree
    )
    
  def fit(self, features:jnp.ndarray, targets:jnp.ndarray):
    """
    Fit
    -----
      Trains the model on the given data. Ideally, the data given should be of JNP format, but any mismatching
      data types will be converted to JNP arrays.
    -----
    Args
    -----
    - features (JNP array) : the features to use
    - targets  (JNP array) : the corresponding targets to use
    """
    #############################################################################################
    #                                  Error pre-checks                                         #
    #############################################################################################
    
    if not self.is_compiled:
      raise RuntimeError("Model is not compiled. Call .compile() first.")
    if not isinstance(features, jnp.ndarray) or not isinstance(targets, jnp.ndarray):
      raise TypeError("features and targets must be JAX NumPy arrays.")
    if features.shape[0] == 0 or targets.shape[0] == 0:
      raise ValueError("features or targets must not be empty.")
    if features.shape[0] != targets.shape[0]:
      raise ValueError("features and targets must have the same number of samples.")
    if len(features) < self.batchsize:
      raise ValueError("batchsize cannot be larger than the number of samples.")
    if features.ndim == 1:
      features = features[:, None]
    if targets.ndim == 1:
      targets = targets[:, None]

    print() if self.verbose >= 1 else None
    
    #############################################################################################
    #                                        Functions                                          #
    #############################################################################################

    @jax.jit
    def propagate(features:jnp.ndarray, parameters:dict) -> tuple[jnp.ndarray, jnp.ndarray]:
      
      activations   = [features]
      weighted_sums = []
      
      x = features
      for i, layer in enumerate(self.layers):
        
        layer_params = parameters.get(f'layer_{i}', {})
        x, weighted_sum = layer.forward(layer_params, x)
        
        activations.append(x)
        weighted_sums.append(weighted_sum)
        
      return {
        'activations': activations,
        'weighted_sums': weighted_sums
      }
    
    def process_batch(params_pytree, opt_state, batch_features, batch_targets, timestep):
      
      activations_and_weighted_sums = propagate(batch_features, params_pytree)
      batch_loss = losses.Loss_calculator.forward_loss(batch_targets, activations_and_weighted_sums['activations'][-1], self.loss, self.regularization[1], self.regularization[0], params_pytree)
      error = self.loss.backward(batch_targets, activations_and_weighted_sums['activations'][-1])

      timestep += 1
      for layer_index in reversed(range(len(self.layers))):
        layer = self.layers[layer_index]
        
        layer_params = params_pytree.get(f'layer_{layer_index}', {})
        
        error, gradients = layer.backward(layer_params, activations_and_weighted_sums['activations'][layer_index], error, activations_and_weighted_sums['weighted_sums'][layer_index])
        
        gradients = losses.Loss_calculator.regularize_grad(layer_params, gradients, self.regularization[1], self.regularization[0], ignore_list=['bias', 'biases'])
        
        if hasattr(layer, "update"):
          params_pytree[f'layer_{layer_index}'], opt_state[f'layer_{layer_index}'] = layer.update(
            self.optimizer,
            self.learning_rate,
            layer_params,
            gradients,
            opt_state[f'layer_{layer_index}'],
            timestep=timestep, 
          )
          
        else:
          continue
        
      return (params_pytree, opt_state), batch_loss
    
    def epoch_batch_step(carry, batch_data):
      
      params, opt_state, accumulated_loss, timestep = carry
      batch_features, batch_targets = batch_data

      (new_params, new_opt_state), batch_loss = process_batch(
        params, 
        opt_state, 
        batch_features, 
        batch_targets,
        timestep,
      )

      return (new_params, new_opt_state, accumulated_loss + batch_loss, timestep + 1), batch_loss
    
    #############################################################################################
    #                                        Variables                                          #
    #############################################################################################
    
    self.is_trained = True

    features, targets = datahandler.split_data(features, targets, 1-self.validation_split)
    validation_features, validation_targets = datahandler.split_data(features, targets, self.validation_split)
    
    callback = self.callback()
    callback.initialization(**locals())
    
    scan_data = datahandler.batch_data(self.batchsize, features, targets)
    
    self.gradients_history = {}
    self.params_history = {}
    
    #############################################################################################
    #                                           Main                                            #
    #############################################################################################
    
    for epoch in (progress_bar(range(self.epochs), "> Training", "Complete", decimals=2, length=50, empty=' ') if self.verbose == 1 else range(self.epochs)):

      callback.before_epoch(**locals())

      (self.params_pytree, self.opt_state, batch_loss, _), _ = jax.lax.scan(
        epoch_batch_step,
        (self.params_pytree, self.opt_state, 0.0, 0), # initial carry
        scan_data
      )
      
      self.gradients_history[f"epoch_{epoch}"] = jax.tree.map(lambda x: x[0], self.opt_state)
      self.params_history[f"epoch_{epoch}"] = self.params_pytree
      
      extra_activations_and_weighted_sums = propagate(validation_features, self.params_pytree) if len(validation_features) > 0 else None
      validation_loss = losses.Loss_calculator.forward_loss(validation_targets, extra_activations_and_weighted_sums['activations'][-1], self.loss, self.regularization[1], self.regularization[0], self.params_pytree) if len(validation_features) > 0 else 0
      
      metric_stats = [metric_fn(validation_targets, extra_activations_and_weighted_sums['activations'][-1]) for metric_fn in self.metrics] if len(self.metrics) > 0 else None
      self.metrics_logs.append(metric_stats)
      
      batch_loss /= self.batchsize
      
      # print(validation_loss)
      
      validation_loss /= self.batchsize
      
      self.error_logs.append(batch_loss)
      self.validation_error_logs.append(validation_loss) if len(validation_features) > 0 else None
      
      ############ post training
      
      callback.after_epoch(**locals())

      if (epoch % self.logging == 0 and self.verbose >= 2) or epoch == 0:
        
        lossROC       = 0 if epoch == 0 else batch_loss      - self.error_logs[epoch-self.logging]
        validationROC = 0 if epoch < self.logging else validation_loss - self.validation_error_logs[epoch-self.logging] if self.validation_split > 0 else 0
        metricROC     = 0 if epoch < self.logging else metric_stats[0] - self.metrics_logs[epoch-self.logging][0] if len(self.metrics) > 0 else 0
        
        prefix = f"\033[1mEpoch {epoch}/{self.epochs}\033[0m ({round( ((epoch)/self.epochs)*100 , 2)}%)"
        prefix += ' ' * (25 + len(f"{self.epochs}") * 2 - len(prefix))
        
        print_loss = f"Loss: {batch_loss:.2E}" if batch_loss > 1000 or batch_loss < 0.0001 else f"Loss: {batch_loss:.4f}"
        print_loss = f"┃ \033[32m{print_loss:16}\033[0m" if lossROC < 0 else f"┃ \033[31m{print_loss:16}\033[0m" if lossROC > 0 else f"┃ {print_loss:16}"
        
        if self.verbose == 2:
          print(prefix + print_loss)
        
        elif self.verbose == 3:
          print_validation = f"V Loss: {validation_loss:.2E}" if validation_loss > 1000 or validation_loss < 0.0001 else f"V Loss: {validation_loss:.4f}" if self.validation_split > 0 else f"V Loss: N/A"
          print_validation = f"┃ \033[32m{print_validation:16}\033[0m" if validationROC < 0 else f"┃ \033[31m{print_validation:16}\033[0m" if validationROC > 0 else f"┃ {print_validation:16}"
          print(prefix + print_loss + print_validation)
        
        elif self.verbose == 4:
          print_metric = f"{self.metrics[0].__class__.__name__}: {metric_stats[0]:.5f}" if len(self.metrics) >= 1 else "Metrics N/A"
          print_metric = f"┃ \033[32m{print_metric:16}\033[0m" if metricROC > 0 else f"┃ \033[31m{print_metric:16}\033[0m" if metricROC < 0 else f"┃ {print_metric:16}"
          
          print_validation = f"V Loss: {validation_loss:.2E}" if validation_loss > 1000 or validation_loss < 0.0001 else f"V Loss: {validation_loss:.4f}" if self.validation_split > 0 else f"V Loss: N/A"
          print_validation = f"┃ \033[32m{print_validation:16}\033[0m" if validationROC < 0 else f"┃ \033[31m{print_validation:16}\033[0m" if validationROC > 0 else f"┃ {print_validation:16}"
          
          print(prefix + print_loss + print_validation + print_metric)
    
    callback.end(**locals())

  def push(self, inputs:jnp.ndarray) -> jnp.ndarray:
    """
    Propagates the input through the entire model, excluding dropout layers (if any).
    weights will not be updated.
    """
    x = inputs
    for i, layer in enumerate(self.layers):
      
      if layer.training_only:
        continue
      
      layer_params = self.params_pytree.get(f'layer_{i}', {})
      x, _ = layer.forward(layer_params, x)
      
    return x

