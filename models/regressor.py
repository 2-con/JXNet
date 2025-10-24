"""
Regression
=====
  Regression algorithms for PyNet with some features from NetCore such as optimizers and losses. Advanced features such as callbacks or
  validation sets are not supported in these models.
-----
Provided Regression Models
-----
- Linear
- Polynomial
- Logistic
- Exponential
- Sinusoidal (External Model)
- Power
- Logarithmic
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import ABC, abstractmethod
import random
import jax, jax.numpy as jnp
from core.standard import initializers, losses, optimizers, callbacks, datahandler
from tools.utility import progress_bar

class Regressor(ABC):
  def __init__(self, input_size, output_size, seed=None):
    self.universal_seed = seed
    if seed is None:
      self.universal_seed = jax.random.PRNGKey(random.randint(0, 2**32))
    
    self.weights = initializers.Default()(seed, (input_size, output_size), input_size, output_size)
    self.biases = initializers.Default()(seed, (output_size,), input_size, output_size)
    self.input_size = input_size
    self.output_size = output_size
    self.is_compiled = False
    
    self.error_logs = []
    self.validation_error_logs = []
  
  def compile(self, loss:losses.Loss, optimizer:optimizers.Optimizer, learning_rate:float, epochs:int, batch_size=1, verbose=1, logging=1, callback=callbacks.Callback, validation_split=0, **kwargs):
    """
    - loss                        (core.standard.losses.Loss)          : loss function to use, not an instance
    - optimizer                   (core.standard.optimizers.Optimizer) : optimizer to use, not an instance
    - learning_rate               (float)                              : learning rate to use
    - epochs                      (int)                                : number of epochs to train for
    
    - (Optional) batch_size       (int)                                : batch size to use
    - (Optional) verbose          (int)                                : verbosity level
    - (Optional) logging          (int)                                : how ofter to report if the verbosity is at least 3
    - (Optional) callbacks        (core.standard.callback)             : call a custom callback class during training with access to all local variables, read more in the documentation.
    - (Optional) validation_split (float)                              : fraction of the data to use for validation, must be between [0, 1). Default is 0 (no validation).
    
    - (Optional) regularization   (tuple[str, float])                  : type of regularization to use, position 0 is the type ("L1" or "L2"), position 1 is the lambda value. Default is None (no regularization).
    
    Verbosity Levels
    -----
    - 0 : None
    - 1 : Progress bar of the whole training process
    - 2 : (Numerical output) Loss
    - 3 : (Numerical output) Loss and V Loss (Validation Loss)
    - 4 : (Numerical output) Loss, V Loss (Validation Loss) and the 1st metric in the 'metrics' list
    """
    self.is_compiled = True
    
    self.loss = loss
    self.optimizer = optimizer
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.batchsize = batch_size
    
    self.verbose = verbose
    self.logging = logging
    self.callback = callback
    self.validation_split = validation_split
    self.regularization = kwargs.get('regularization', ["None", 1])
    
    if verbose < 0 or verbose > 4:
      raise ValueError("Verbosity level must be between 0 and 4.")
    if type(verbose) != int:
      raise TypeError("Verbosity level must be an integer.")
    if type(learning_rate) != float:
      raise TypeError("Learning rate must be a float.")
    if type(epochs) != int:
      raise TypeError("Epochs must be an integer.")
    if not (0 <= validation_split < 1):
      raise ValueError("validation_split must be between [0,1)")
    
    self.opt_state = self.optimizer.initialize((self.output_size, self.input_size), jnp.float32)

  @abstractmethod
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
    
    def process_batch(params_pytree, opt_state, batch_features, batch_targets, timestep):
      
      activations_and_weighted_sums = self.predict(batch_features, params_pytree)
      epoch_loss = losses.Loss_calculator.forward_loss(batch_targets, activations_and_weighted_sums['activations'][-1], self.loss, self.regularization[1], self.regularization[0], params_pytree)
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
        
      return (params_pytree, opt_state), epoch_loss
    
    def epoch_batch_step(carry, batch_data):
      
      params, opt_state, accumulated_loss, timestep = carry
      batch_features, batch_targets = batch_data

      (new_params, new_opt_state), epoch_loss = process_batch(
        params, 
        opt_state, 
        batch_features, 
        batch_targets,
        timestep,
      )

      return (new_params, new_opt_state, accumulated_loss + epoch_loss, timestep + 1), epoch_loss
    
    #############################################################################################
    #                                        Variables                                          #
    #############################################################################################
    
    self.is_trained = True

    features, targets = datahandler.split_data(features, targets, 1-self.validation_split)
    validation_features, validation_targets = datahandler.split_data(features, targets, self.validation_split)
    
    self.callback.initialization(**locals())
    
    scan_data = datahandler.batch_data(self.batchsize, features, targets)
    
    self.gradients_history = {}
    self.params_history = {}
    
    #############################################################################################
    #                                           Main                                            #
    #############################################################################################
    
    for epoch in (progress_bar(range(self.epochs), "> Training", "Complete", decimals=2, length=50, empty=' ') if self.verbose == 1 else range(self.epochs)):

      self.callback.before_epoch(**locals())

      (self.params_pytree, self.opt_state, epoch_loss, _), _ = jax.lax.scan(
        epoch_batch_step,
        (self.params_pytree, self.opt_state, 0.0, 0), # initial carry
        scan_data
      )
      
      self.gradients_history[f"epoch_{epoch}"] = jax.tree.map(lambda x: x[0], self.opt_state)
      self.params_history[f"epoch_{epoch}"] = self.params_pytree
      
      extra_activations_and_weighted_sums = self.predict(validation_features, self.params_pytree) if len(validation_features) > 0 else None
      validation_loss = losses.Loss_calculator.forward_loss(validation_targets, extra_activations_and_weighted_sums['activations'][-1], self.loss, self.regularization[1], self.regularization[0], self.params_pytree) if len(validation_features) > 0 else 0
      
      epoch_loss /= self.batchsize
      
      # print(validation_loss)
      
      validation_loss /= self.batchsize
      
      self.error_logs.append(epoch_loss)
      self.validation_error_logs.append(validation_loss) if len(validation_features) > 0 else None
      
      ############ post training
      
      self.callback.after_epoch(**locals())

      if (epoch % self.logging == 0 and self.verbose >= 2) or epoch == 0:
        
        lossROC       = 0 if epoch == 0 else epoch_loss      - self.error_logs[epoch-self.logging]
        validationROC = 0 if epoch < self.logging else validation_loss - self.validation_error_logs[epoch-self.logging] if self.validation_split > 0 else 0
        
        prefix = f"\033[1mEpoch {epoch}/{self.epochs}\033[0m ({round( ((epoch)/self.epochs)*100 , 2)}%)"
        prefix += ' ' * (25 + len(f"{self.epochs}") * 2 - len(prefix))
        
        print_loss = f"Loss: {epoch_loss:.2E}" if epoch_loss > 1000 or epoch_loss < 0.0001 else f"Loss: {epoch_loss:.4f}"
        print_loss = f"┃ \033[32m{print_loss:16}\033[0m" if lossROC < 0 else f"┃ \033[31m{print_loss:16}\033[0m" if lossROC > 0 else f"┃ {print_loss:16}"
        
        if self.verbose == 2:
          print(prefix + print_loss)
        
        elif self.verbose == 3:
          print_validation = f"V Loss: {validation_loss:.2E}" if validation_loss > 1000 or validation_loss < 0.0001 else f"V Loss: {validation_loss:.4f}" if self.validation_split > 0 else f"V Loss: N/A"
          print_validation = f"┃ \033[32m{print_validation:16}\033[0m" if validationROC < 0 else f"┃ \033[31m{print_validation:16}\033[0m" if validationROC > 0 else f"┃ {print_validation:16}"
          print(prefix + print_loss + print_validation)
        
    self.callback.end(**locals())
  
  @abstractmethod
  def predict(self, point):
    pass
