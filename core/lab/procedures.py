"""
Procedures
=====
  Procedures are actions Netlab can perform on a StandardNet model and are performed sequentially.
  
Proviedes:
- Procedure
  - Base class all JXNet procedures must inherit from

- Compile
- Train
- Evaluate
- Values
- Track Layers
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from abc import ABC, abstractmethod
from standard.metrics import Metric

"""
TODO: add conditonals
TODO: add checkpoints
"""

class Procedure(ABC):
  """
  Base class for procedures to be applied to a StandardNet model
  
  A Procedure class is required to have the following:
  
  - A '__init__' method with any constant object attributes should be defined here
    - Args:
      - model     (StandardNet model) : the model to apply the spesific procedure to, must be compiled
      - *args     (any)               : any arguments to be passed to the procedure
      - **kwargs  (any)               : any keyword arguments to be passed to the procedure
    - Returns:
      - None
  
  - A '__call__' method that applies the procedure to the model
    - Args:
      - iteration (int)               : the current cycle of the experiment
      - *args     (any)               : any arguments to be passed to the procedure
      - **kwargs  (any)               : any keyword arguments to be passed to the procedure
    - Returns:
      - dict : anything that needs to be logged
  """

  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs
  
  @abstractmethod
  def __call__(self, *args, **kwargs):
    return {}

##########################################################################################################
#                                            Built-in Contents                                           #
##########################################################################################################

# Model-spesific Procedures

class Compile(Procedure):
  """
    Compile
    -----
      Recompiles the model to be ready for training, completely resets all parameters and states of the model. 
      This step is required to empty the history of the model.
    -----
    Args
    -----
    - args   (any) : any arguments to be passed to the compile method of the model
    - kwargs (any) : any keyword arguments to be passed to the compile method of the model
    """
  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs
  
  def __call__(self, data, model, cycle, *args, **kwargs):
    
    newargs = []
    for arg in self.args:
      if isinstance(arg, Values):
        newargs.append( arg(cycle=cycle) )
      else:
        newargs.append(arg)
    
    newkwargs = {}
    for key, value in self.kwargs.items():
      if isinstance(value, Values):
        newkwargs[key] = value(cycle=cycle)
      else:
        newkwargs[key] = value
    
    model.compile(*newargs, **newkwargs)
    return "recompiled"
  
class Train(Procedure):
  """
  Train
  -----
    Trains the model on the given data. Ideally, the data given should be of JNP format, but any mismatching data will not be checked
    since this procedure only forward information to the model.
  -----
  Args
  -----
  - features (String) : the features to use from the dataset
  - targets  (String) : the corresponding targets to use from the dataset
  """
  def __init__(self, feature_name:str, target_name:str):
    self.feature_name = feature_name
    self.target_name = target_name
    
    if isinstance(self.feature_name, Values) or isinstance(self.target_name, Values):
      raise ValueError("Feature name and target name must be strings, not Values procedure.")
  
  def __call__(self, data, model, cycle, *args, **kwargs):
    model.fit(data[self.feature_name], data[self.target_name])
    return "trained"

class Evaluate(Procedure):
  """
  Evaluate
  -----
    Evaluates the model on the given data. Ideally, the data given should be of JNP format, but any mismatching data will not be checked
    since this procedure only forward information to the model.
  -----
  Args
  -----
  - features (String)                     : the features to use from the dataset
  - targets  (String)                     : the corresponding targets to use from the dataset
  - loss     (JXNet Loss or JXNet Metric) : the loss function to use
  """
  def __init__(self, feature_name:str, target_name:str, metric_function:Metric):
    self.feature_name = feature_name
    self.target_name = target_name
    self.metric_function = metric_function
    
    if isinstance(self.feature_name, Values) or isinstance(self.target_name, Values):
      raise ValueError("Feature name and target name must be strings, not Values procedure.")
  
  def __call__(self, data, model, cycle, *args, **kwargs):
    return self.metric_function.forward(model.push(data[self.feature_name]), data[self.target_name])

# Analysis Procedures
  
class Values(Procedure):
  def __init__(self, *values):
    """
    Values
    -----
      A procedure that returns a constant at each cycle.
      if the cycle exceed the number of values, it will return None. However if the cycle is less than
      the number of values, it will return the value at that cycle regardless if it is the last cycle.
    -----
    Args
    -----
    - values  (any) : any number of constant values to be returned at each cycle
    """
    self.values = values
    pass
  
  def __call__(self, data, model, cycle, *args, **kwargs):
    return self.values[cycle] if cycle < len(self.values) else None

class Track_Layer(Procedure):
  """
  Track Layer
  -----
    Tracks a layer of the model and returns the parameters and gradients of the layer. The formating for the path follows [Layer_{index}, {parameter name}]
    to properly track the parameters and gradients of the model.
  -----
  Args
  -----
  - datapath (tuple[str, str]) : the datapath toward the values to track
  """
  def __init__(self, datapath:tuple[str,str]):
    self.datapath = datapath
  
  def __call__(self, data, model, cycle, *args, **kwargs):
    
    parameters = {
      epoch : per_epoch_history.get(self.datapath[0](cycle) if isinstance(self.datapath[0], Values) else self.datapath[0], {}).get(self.datapath[1](cycle) if isinstance(self.datapath[1], Values) else self.datapath[1], [])
      for epoch, per_epoch_history in model.params_history.items()
    }
    
    gradients = {
      epoch : per_epoch_history.get(self.datapath[0](cycle) if isinstance(self.datapath[0], Values) else self.datapath[0], {}).get(self.datapath[1](cycle) if isinstance(self.datapath[1], Values) else self.datapath[1], [])
      for epoch, per_epoch_history in model.gradients_history.items()
    }
    
    return {"gradients":gradients, "parameters":parameters}


