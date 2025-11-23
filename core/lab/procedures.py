import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from abc import ABC, abstractmethod
import copy
import jax

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
  def __call__(self, model, cycle_index, *args, **kwargs):
    return {}

##########################################################################################################
#                                            Built-in Contents                                           #
##########################################################################################################

class Values(Procedure):
  def __init__(self, *values):
    """
    Values
    -----
      A procedure that returns a set of constant values at each cycle.
      if the cycle exceed the number of values, it will return None. However if the cycle is less than
      the number of values, it will return the value at that cycle regardless if it is the last cycle.
    -----
    Args
    -----
    - values  (any) : any number of constant values to be returned at each cycle
    """
    self.values = values
    pass
  
  def __call__(self, cycle_index, *args, **kwargs):
    return self.values[cycle_index] if cycle_index < len(self.values) else None

class Compile(Procedure):
  def __init__(self, *args, **kwargs):
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
    self.args = args
    self.kwargs = kwargs
  
  def __call__(self, model, data, verbose, cycle_index, *args, **kwargs):
    
    newargs = []
    for arg in self.args:
      if isinstance(arg, Values):
        newargs.append( arg(cycle_index=cycle_index) )
      else:
        newargs.append(arg)
    
    newkwargs = {}
    for key, value in self.kwargs.items():
      if isinstance(value, Values):
        newkwargs[key] = value(cycle_index=cycle_index)
      else:
        newkwargs[key] = value
    
    model.compile(*newargs, **newkwargs)
    return "recompiled"
  
class Train(Procedure):
  def __init__(self, feature_name:str, target_name:str):
    self.feature_name = feature_name
    self.target_name = target_name
    
    if isinstance(self.feature_name, Values) or isinstance(self.target_name, Values):
      raise ValueError("Feature name and target name must be strings, not Values procedure.")
  
  def __call__(self, model, data, verbose, cycle_index, *args, **kwargs):
    model.fit(data[self.feature_name], data[self.target_name])
    return "trained"
  
class Track_Layer(Procedure):
  def __init__(self, datapath:tuple[str,str]):
    self.datapath = datapath
  
  def __call__(self, model, data, verbose, cycle_index, *args, **kwargs):
    
    parameters = {
      epoch : per_epoch_history.get(self.datapath[0](cycle_index) if isinstance(self.datapath[0], Values) else self.datapath[0], {}).get(self.datapath[1](cycle_index) if isinstance(self.datapath[1], Values) else self.datapath[1], [])
      for epoch, per_epoch_history in model.params_history.items()
    }
    
    gradients = {
      epoch : per_epoch_history.get(self.datapath[0](cycle_index) if isinstance(self.datapath[0], Values) else self.datapath[0], {}).get(self.datapath[1](cycle_index) if isinstance(self.datapath[1], Values) else self.datapath[1], [])
      for epoch, per_epoch_history in model.gradients_history.items()
    }
    
    return {"gradients":gradients, "parameters":parameters}


