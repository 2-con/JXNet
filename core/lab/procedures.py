import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from abc import ABC, abstractmethod
import copy

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

  @abstractmethod
  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs
  
  @abstractmethod
  def __call__(self, model, timestep, *args, **kwargs):
    return {}

class Track_Layer(Procedure):
  def __init__(self, datapath:tuple[str,str], *args, **kwargs):
    self.datapath = datapath
    self.args = args
    self.kwargs = kwargs
  
  def __call__(self, model, iteration, *args, **kwargs):
    gradients = self.model.gradients_history[self.datapath[0]][self.datapath[1]]
    parameters = self.model.params_history[self.datapath[0]][self.datapath[1]]
    
    return {"gradients":gradients, "parameters":parameters}

class Ablate(Procedure):
  def __init__(self, targer_layer_indecies:tuple[int,...], *args, **kwargs):
    """
    Standardnet:recompile
    Staticnet:ok
    """ 
    self.indecises = targer_layer_indecies
    self.args = args
    self.kwargs = kwargs
  
  def __call__(self, model, timestep, *args, **kwargs):
    temporary_layers = copy.deepcopy(model.layers)
    temporary_layers.pop( self.indecises[timestep % len(self.indecises)] )
    self.model.layers = temporary_layers

    return {"ablated layer": self.model.layers[self.indecises[timestep % len(self.indecises)]].__class__.__name__}
    
