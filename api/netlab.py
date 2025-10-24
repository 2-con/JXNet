"""
NetLab API
=====
  A high-level API for experimenting, benchmarking and testing both StandardNet and StaticNet models.
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
from core.lab.procedures import Procedure

class Sample:
  def __init__(self, *args):
    """
    Sample
    -----
      Defines a sample set of models to be benchmarked listed in order of evaluation.
    """
    self.models = args
    
    self.logs = {}
  
  def procedure(self, *args:Procedure):
    """
    Procedure
    -----
      Defines a set of procedures to be applied to the models listed in order of evaluation. This procedure is applied once per cycle, and any custom procedure(s)
      passed should not leak state between cycles, even if its for the same model.
    """
    self.procedures = args
    
    if any([not isinstance(procedure, Procedure) for procedure in self.procedures]):
      raise ValueError("All procedures must be of type Procedure or inherit from it. It can be found in core > lab > procedures.py")
    
  def compile(self, cycles:int, verbose:int=0, logging:int=1, name:str="Experiment", *args, **kwargs):
    """
    Compile
    -----
      Compiles the sample to be ready for benchmarking.
    -----
    Args
    -----
    - cycles                      (int) : number of times to run the experiment
    - (Optional) verbose          (int) : verbosity level
    - (Optional) logging          (int) : how often to report if the verbosity is at least 3
    - (Optional) name             (str) : name of the experiment, defaults to "Experiment"
    
    Verbosity Levels
    -----
    - 0 : None
    - 1 : Progress bar of the whole experiment
    - 2 : Progress bar of each cycle
    """
    self.name = name
    self.cycles = cycles
    self.logging = logging
    self.verbose = verbose
    
    if cycles <= 0:
      raise ValueError("Cycles must be greater than 0.")
    for index, model in enumerate(self.models):
      if model.__api__ == "StaticNet":
        print(f"NetLab detected a StaticNet model in sample {index+1}. StaticNet models cannot be tinkered comprehensively and are mostly unsupported for anything beyond observation and compilation-config changes.")
      elif model.__api__ == "StandardNet":
        pass
      else:
        print(f"NetLab detected an unrecognized model API in sample {index+1}. Please make sure that your model is compatible with the procedures given.")
  
  def run(self, dataset):
    """
    Run
    -----
      Runs the experiment on the given dataset.
    -----
    Args
    -----
    - dataset (dict) : the labeled dataset to run the experiment on
    """
    # data > model > cycle > procedures
    all_logs = {}
    
    # cydle through the datasets
    for dataset_name, data in dataset.items():
      all_logs[dataset_name] = {}
      
      # cydle through the models
      for index, model in enumerate(self.models):
        all_logs[dataset_name][f"Model {index+1}"] = {}
        
        # cydle through the cycle
        for cycle in range(self.cycles):
          all_logs[dataset_name][f"Model {index+1}"][f"Cycle {cycle+1}"] = []
          
          # cydle through the procedures
          for procedure in self.procedures:
            
            # run the procedure and log results
            all_logs[dataset_name][f"Model {index+1}"][f"Cycle {cycle+1}"][f"Procedure {procedure.__name__}"] = procedure(
              model, data, self.verbose, self.logging, self.name, cycle, dataset_name, index
            )

    
    
    self.logs = all_logs