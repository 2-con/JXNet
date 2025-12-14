"""
Regression
=====
  Regression algorithms for JXNet with some features such as optimizers and losses. Advanced features such as callbacks or
  validation sets are not supported in these models.

Provides:
- Regressor
  - A base class for all regression models to inherit. Mainly contain the bulk of the code for the models, the only thing
    needed to create a regressor is the forward and backward pass functions. The update and data iteration is alredy handled.

- Linear
- Polynomial
- Logistic
- Power
- Exponential
- Logarithmic
"""

from abc import ABC, abstractmethod
import random
import jax, jax.numpy as jnp
from jxnet.core.standard import initializers, losses, optimizers, callbacks, datahandler, functions
from jxnet.tools.utility import progress_bar
import itertools
import math

class Regressor(ABC):
  """
  Regressior
  -----
    Base class for regression models
  -----
  """
  parameters = {
    "weights": ("input_size", "output_size"),
    "biases": ("output_size",),
  }
  
  def __init__(self, input_size, output_size):
    self.input_size = input_size
    self.output_size = output_size
    self.is_compiled = False
  
  def compile(self, loss:losses.Loss, optimizer:optimizers.Optimizer, learning_rate:float, epochs:int, initializer=initializers.Default(), verbose=1, logging=1, callback=callbacks.Callback, validation_split=0, seed=None, **kwargs):
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
    """
    self.is_compiled = True
    
    self.loss = loss
    self.optimizer = optimizer
    self.learning_rate = learning_rate
    self.epochs = epochs
    
    self.verbose = verbose
    self.logging = logging
    self.callback = callback()
    self.validation_split = validation_split
    self.universal_seed = seed
    self.regularization = kwargs.get('regularization', ["None", 1])
    self.error_logs = []
    self.validation_error_logs = []
    
    self.gradients_history = {}
    self.params_history = {}
    
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
    
    if seed is None:
      self.universal_seed = random.randint(0, 2**32)
    
    self.params = { 
      key: initializer(self.universal_seed, tuple(getattr(self, attr) for attr in shape), self.input_size, self.output_size) 
      for key, shape in self.parameters.items()
    }
    
    self.opt_state = jax.tree.map(
      lambda p: self.optimizer.initialize(p.shape, p.dtype),
      self.params
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
    if features.ndim == 1:
      features = features[:, None]
    if targets.ndim == 1:
      targets = targets[:, None]

    print() if self.verbose >= 1 else None
    
    #############################################################################################
    #                                        Functions                                          #
    #############################################################################################
    
    def update(optimizer, learning_rate, params:dict, gradients:jnp.ndarray, opt_state:dict, *args, **kwargs) -> dict:
      updated_params = {}
      new_opt_state = {}
      
      for name, value in params.items():
        updated_params[name], new_opt_state[name] = optimizer.update(
          learning_rate,
          value,
          gradients[name],
          opt_state[name],
          *args, 
          **kwargs
        )
      
      return updated_params, new_opt_state
    
    def process_batch(params, opt_state, batch_features, batch_targets, timestep):
      
      activations, weighted_sums = self.forward(batch_features, params)
      epoch_loss = losses.Loss_calculator.regressor_forward_loss(batch_targets, activations, self.loss, self.regularization[1], self.regularization[0], params)
      error = self.loss.backward(batch_targets, activations)
      
      gradients = self.backward(activations, error, weighted_sums)
      gradients = losses.Loss_calculator.regularize_grad(params, gradients, self.regularization[1], self.regularization[0], ignore_list=['bias'])
      
      new_params, new_opt_state = update(
        self.optimizer,
        self.learning_rate,
        params,
        gradients,
        opt_state,
        timestep=timestep, 
      )
        
      return (new_params, new_opt_state), epoch_loss
    
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

    validation_features, validation_targets = datahandler.split_data(features, targets, self.validation_split)
    train_features, train_targets = datahandler.split_data(features, targets, 1-self.validation_split)
    
    self.callback.initialization(**locals())
    timestep = 1
    batchsize = train_features.shape[0]
    
    scan_data = datahandler.batch_data(batchsize, train_features, train_targets)
    
    #############################################################################################
    #                                           Main                                            #
    #############################################################################################
    
    for epoch in (progress_bar(range(self.epochs), "> Training", "Complete", decimals=2, length=50, empty=' ') if self.verbose == 1 else range(self.epochs)):

      self.callback.before_epoch(**locals())

      (self.params, self.opt_state, epoch_loss, timestep), _ = jax.lax.scan(
        epoch_batch_step,
        (self.params, self.opt_state, 0.0, timestep), # initial carry
        scan_data
      )
      
      self.gradients_history[f"epoch_{epoch}"] = jax.tree.map(lambda x: x.copy(), self.opt_state)
      self.params_history[f"epoch_{epoch}"] = self.params
      
      activations = self.predict(validation_features) if len(validation_features) > 0 else 0,0
      validation_loss = losses.Loss_calculator.forward_loss(validation_targets, activations, self.loss, self.regularization[1], self.regularization[0], self.params) if len(validation_features) > 0 else 0
      
      epoch_loss /= train_features.shape[0]
      validation_loss /= batchsize
      
      self.error_logs.append(epoch_loss)
      self.validation_error_logs.append(validation_loss) if len(validation_features) > 0 else None
      
      ############ post training
      
      self.callback.after_epoch(**locals())

      if (epoch % self.logging == 0 or epoch == 0) and self.verbose >= 2 :
        
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
  def forward(self, inputs:jnp.ndarray, params:dict) -> tuple[jnp.ndarray, jnp.ndarray]:
    return None, None
  
  @abstractmethod
  def backward(self, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> tuple[jnp.ndarray, dict]:
    return {None: None,}
  
  def predict(self, inputs:jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    return self.forward(inputs, self.params)[0]

class Linear(Regressor):
  """
  Linear Regression
  -----
    Fits a linear model in the form y = a * x + b to a dataset.
  -----
  Args
  -----
  - input_size  (int) : input dimension
  - output_size (int) : output dimension
  """
  @staticmethod
  def forward(inputs:jnp.ndarray, params:dict) -> tuple[jnp.ndarray, jnp.ndarray]:
    # inputs: (batch, in_features), weights: (in_features, out_features)
    weighted_sums = inputs @ params['weights'] + params['biases']
    activated_output = functions.Identity().forward(weighted_sums)
    return activated_output, weighted_sums
  
  @staticmethod
  def backward(inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> dict:
    # error: (batch, out_features), inputs: (batch, in_features)
    
    grads_z = functions.Identity().backward(error, weighted_sums)['x']
    
    grads_weights = jnp.einsum("bi,bj->ij", inputs, grads_z)  # (in_features, out_features)
    grads_biases = jnp.sum(grads_z, axis=0)  # (out_features,)

    return {'weights': grads_weights, 'biases': grads_biases,}

class Polynomial(Regressor):
  """
  Polynomial Regression
  -----
    Fits a Polynomial model in the form y = a * x^n + b * x^(n-1) + ... + C to a dataset.
    Ensure that the degree is non-negative as JXNet does not provide safety checks by design.
  -----
  Args
  -----
  - degree      (int) : degree of the polynomial
  - input_size  (int) : input dimension
  - output_size (int) : output dimension
  """
  def __init__(self, degree:int, input_size:int, output_size:int):
    super().__init__(input_size, output_size)
    self.degree = degree
    self.input_size = math.comb(self.input_size + degree, degree)
  
  @staticmethod
  def forward(inputs:jnp.ndarray, params:dict) -> tuple[jnp.ndarray, jnp.ndarray]:
    
    def poly_features(input_values: jnp.ndarray, degree: int) -> jnp.ndarray:
      num_original_features = input_values.shape[-1]
      
      polynomial_features_list = []
      for d in range(1, degree + 1):
        for combo_indices in itertools.combinations_with_replacement(range(num_original_features), d):
          # Calculate the product using JNP (this is the core array operation)
          term_value = jnp.prod(input_values[jnp.array(combo_indices)])
          polynomial_features_list.append(term_value)

      return jnp.array(polynomial_features_list)
    
    # inputs: (batch, in_features), weights: (in_features, out_features)
    weighted_sums = poly_features(inputs) @ params['weights'] + params['biases']
    activated_output = functions.Identity().forward(weighted_sums)
    return activated_output, weighted_sums
  
  @staticmethod
  def backward(inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> dict:
    # error: (batch, out_features), inputs: (batch, in_features)
    
    grads_z = functions.Identity().backward(error, weighted_sums)['x']
    
    grads_weights = jnp.einsum("bi,bj->ij", inputs, grads_z)  # (in_features, out_features)
    grads_biases = jnp.sum(grads_z, axis=0)  # (out_features,)

    return {'weights': grads_weights, 'biases': grads_biases,}

class Logistic(Regressor):
  """
  Logistic Regression
  -----
    Fits a Logistic model in the form y = Sigmoid(a * x + b) to a dataset.
  -----
  Args
  -----
  - input_size  (int) : input dimension
  - output_size (int) : output dimension
  """
  @staticmethod
  def forward(inputs:jnp.ndarray, params:dict) -> tuple[jnp.ndarray, jnp.ndarray]:
    # inputs: (batch, in_features), weights: (in_features, out_features)
    weighted_sums = inputs @ params['weights'] + params['biases']
    activated_output = functions.Sigmoid().forward(weighted_sums)
    return activated_output, weighted_sums
  
  @staticmethod
  def backward(inputs, error, weighted_sums) -> dict:
    # error: (batch, out_features), inputs: (batch, in_features)
    
    grads_z = functions.Sigmoid().backward(error, weighted_sums)['x']
    
    grads_weights = jnp.einsum("bi,bj->ij", inputs, grads_z)  # (in_features, out_features)
    grads_biases = jnp.sum(grads_z, axis=0)  # (out_features,)

    return {'weights': grads_weights, 'biases': grads_biases,}

class Power(Regressor):
  """
  Power Regression
  -----
    Fits a power function in either log mode (y = log(a) + x * log(b)) or non-log mode (y = a * x^b).
    Make sure that none of the inputs are negative when using log mode.
  -----
  Args
  -----
  - log_mode    (bool) : whether to transform the model to a log-log space
  - input_size  (int)  : input dimension
  - output_size (int)  : output dimension
  """
  def __init__(self, log_mode:bool, input_size:int, output_size:int):
    super().__init__(input_size, output_size)
    self.log_mode = log_mode
  
  def forward(self, inputs:jnp.ndarray, params:dict) -> tuple[jnp.ndarray, jnp.ndarray]:
    
    # W is 'b', B is 'log(a)' if log_mode=True, or 'a' if log_mode=False
    W = params['weights']
    B = params['biases']
    
    EPSILON = 1e-8
    
    if self.log_mode:
      
      log_inputs = jnp.log(inputs)
      weighted_sums = log_inputs @ W + B
      
      activated_output = jnp.exp(weighted_sums)
      
      return activated_output, weighted_sums
    
    else: # Non-log mode: y = B * x^W
      if inputs.min() <= 0 and not jnp.all(jnp.round(W) == W):
        raise ValueError("Non-log mode requires positive inputs if exponent (weights) is not an integer.")

      weighted_sums = B[0] * (inputs + EPSILON)**W
      
      activated_output = weighted_sums
      
      return activated_output, weighted_sums

  def backward(self, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> dict:
    if inputs.ndim == 1: inputs = inputs[:, None]
    
    # dL/dy_pred = error (from loss.backward)
    dL_dy = functions.Identity().backward(error, weighted_sums)['x']
    
    W = self.params['weights']
    B = self.params['biases']
    
    EPSILON = 1e-8
    
    if self.log_mode:
      dL_dlogy = dL_dy * weighted_sums
      log_inputs = jnp.log(inputs)
      
      grads_z = dL_dy * jnp.exp(weighted_sums)
      
      grads_weights = jnp.einsum("bi,bj->ij", log_inputs, grads_z)
      grads_biases = jnp.sum(grads_z, axis=0)
      
      return {'weights': grads_weights, 'biases': grads_biases,}
    
    else: # Non-log mode: y = B * x^W
      x_W = (inputs + EPSILON)**W
      
      grads_weights = dL_dy * (B[0] * x_W * jnp.log(inputs + EPSILON))
      grads_weights = jnp.sum(grads_weights, axis=0)
      
      grads_biases = dL_dy * x_W
      grads_biases = jnp.sum(grads_biases, axis=0)
      
      return {'weights': grads_weights, 'biases': grads_biases,}

class Exponential(Regressor):
  """
  Exponential Regression
  -----
    Fits an exponential function in either log-linear mode (log(y) = log(a) + c * x) or 
    non-log mode (y = a * exp(c * x)).
    Make sure that none of the target values (y) are non-positive when using log mode.
  -----
  Args
  -----
  - log_mode      (bool) : whether to transform the model to a log-linear space
  - input_size    (int)  : input dimension
  - output_size   (int)  : output dimension
  """
  def __init__(self, log_mode:bool, input_size:int, output_size:int):
    super().__init__(input_size, output_size)
    self.log_mode = log_mode
  
  def forward(self, inputs:jnp.ndarray, params:dict) -> tuple[jnp.ndarray, jnp.ndarray]:

    W = params['weights']
    B = params['biases']
    
    if self.log_mode:
      
      weighted_sums = inputs @ W + B
      activated_output = jnp.exp(weighted_sums)
      
      return activated_output, weighted_sums
    
    else:
      exp_term = jnp.exp(inputs @ W)
      weighted_sums = B[0] * exp_term
      
      return weighted_sums, weighted_sums # activation is the same as WS

  def backward(self, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> dict:
    dL_dy = functions.Identity().backward(error, weighted_sums)['x']
    
    W = self.params['weights']
    B = self.params['biases']
    
    if self.log_mode:
      
      grads_z = dL_dy * jnp.exp(weighted_sums)
      grads_weights = jnp.einsum("bi,bj->ij", inputs, grads_z)
      
      grads_biases = jnp.sum(grads_z, axis=0)
      
      return {'weights': grads_weights, 'biases': grads_biases,}
    
    else: # Normal mode: y = B[0] * exp(W * x)
      
      linear_combo = inputs @ W
      exp_term = jnp.exp(linear_combo) # exp(c*x)
      
      grads_biases = dL_dy * exp_term
      grads_biases = jnp.sum(grads_biases, axis=0)
      
      grads_weights = dL_dy * (weighted_sums * inputs)
      grads_weights = jnp.sum(grads_weights, axis=0)
      
      return {'weights': grads_weights, 'biases': grads_biases,}

class Logarithmic(Regressor):
  """
  Logarithmic Regression
  -----
    Fits a Logarithmic model in the form y = a + b * log(x) to a dataset.
  -----
  Args
  -----
  - input_size  (int) : input dimension
  - output_size (int) : output dimension
  """
  def forward(self, inputs:jnp.ndarray, params:dict) -> tuple[jnp.ndarray, jnp.ndarray]:
    if inputs.min() <= 0:
      raise ValueError("Input values must be greater than 0 for Logarithmic Regressor.")

    log_inputs = jnp.log(inputs)

    weighted_sums = log_inputs @ params['weights'] + params['biases']
    activated_output = functions.Identity().forward(weighted_sums)
    
    return activated_output, weighted_sums

  def backward(self, inputs:jnp.ndarray, error:jnp.ndarray, weighted_sums:jnp.ndarray) -> dict:
    log_inputs = jnp.log(inputs)
    
    grads_z = functions.Identity().backward(error, weighted_sums)['x']
    
    grads_weights = jnp.einsum("bi,bj->ij", log_inputs, grads_z)
    grads_biases = jnp.sum(grads_z, axis=0)
    
    return {'weights': grads_weights, 'biases': grads_biases,}
