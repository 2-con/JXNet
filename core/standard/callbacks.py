"""
Callbacks
=====
  A callback is a function that is called during the training process once per epoch. A comprehensive analysis of JXNet code is needed to understand
  the names of desired variables.
-----
Provides
-----
- Callback
  - The base class all JXNet callbacks must inherit and follow.
    Contains essencial scaffolding for custom and built-in layers since StandardNet calls all callbacks once per epoch.
    Note that not all callback methods have to be called, leaving callbacks empty is acceptable.

- Loss_Plotter
"""

import matplotlib.pyplot as plt
import jax

class Callback:
  """
  Callable
  -----
    A callback is a function that is called during the training process. It is advised to create a custom callback and inherit from this class since 
    there are some essential methods that need to be present.
  -----
  Args
  -----
  - __init__       (function) : a callback instance is created before training, no arguments are passed
  - initialization (function) : called only once after the callback instance is created
  - before_epoch   (function) : called once at the start of each epoch
  - before_update  (function) : called once before backpropagation and update
  - after_update   (function) : called once after backpropagation and update
  - after_epoch    (function) : called once at the end of each epoch
  - end            (function) : called once at the end of training
  """
  def __init__(callbackself):
    pass
  
  def initialization(callbackself, *args, **kwargs):
    pass
  
  def before_epoch(callbackself, *args, **kwargs):
    pass
  
  def after_update(callbackself, *args, **kwargs):
    pass
  
  def after_epoch(callbackself, *args, **kwargs):
    pass
  
  def end(callbackself, *args, **kwargs):
    pass

##########################################################################################################
#                                            Built-in Contents                                           #
##########################################################################################################

class LossPlotter(Callback):
  """
  Loss Plotter
  -----
    A callback that plots the training loss in real-time during training. Just place this premade callback in the compile method of the model.
    This callback assumes that validation_split is used during training.
  """
  def __init__(callbackself):
    callbackself.fig = None
    callbackself.ax = None
    callbackself.lines = {}
      
  def initialization(callbackself, *args, **kwargs):
    plt.ion()
    callbackself.fig, callbackself.ax = plt.subplots()
    callbackself.ax.set_xlabel('Epoch')
    callbackself.ax.set_ylabel('Training Loss')
    callbackself.ax.set_yscale('log')
    callbackself.ax.grid(True)
    
    # Create line objects for training and validation loss
    callbackself.lines['train'], = callbackself.ax.plot([], [])
      
  def after_epoch(callbackself, *args, **kwargs):
    
    x_data = list(range(kwargs.get('epoch', 0) + 1))
    y_data = kwargs.get('self', None).error_logs
    
    callbackself.lines['train'].set_data(x_data, y_data)
    
    # Dynamically adjust plot limits
    callbackself.ax.set_xlim(0, max(x_data) + 1 if x_data else 1)
    callbackself.ax.set_ylim(min(y_data) / 1.01, max(y_data) * 1.01)
    
    # Draw and flush events to update the plot
    callbackself.fig.canvas.draw()
    callbackself.fig.canvas.flush_events()
      
  def end(*args, **kwargs):
    plt.ioff()
    plt.show()

class ValidationPlotter(Callback):
  """
  Validation Plotter
  -----
    A callback that plots the  validation loss in real-time during training. Just place this premade callback in the compile method of the model.
    This callback assumes that validation_split is used during training.
  """
  def __init__(callbackself):
    callbackself.fig = None
    callbackself.ax = None
    callbackself.lines = {}
      
  def initialization(callbackself, *args, **kwargs):
    plt.ion()
    callbackself.fig, callbackself.ax = plt.subplots()
    callbackself.ax.set_xlabel('Epoch')
    callbackself.ax.set_ylabel('Validation Loss')
    callbackself.ax.set_yscale('log')
    callbackself.ax.grid(True)
    
    # Create line objects for training and validation loss
    callbackself.lines['validation'], = callbackself.ax.plot([], [])
      
  def after_epoch(callbackself, *args, **kwargs):
    
    x_data = list(range(kwargs.get('epoch', 0) + 1))
    y_data = kwargs.get('self', None).validation_error_logs
    
    callbackself.lines['validation'].set_data(x_data, y_data)
    
    # Dynamically adjust plot limits
    callbackself.ax.set_xlim(0, max(x_data) + 1 if x_data else 1)
    callbackself.ax.set_ylim(min(y_data) / 1.01, max(y_data) * 1.01)
    
    # Draw and flush events to update the plot
    callbackself.fig.canvas.draw()
    callbackself.fig.canvas.flush_events()
      
  def end(*args, **kwargs):
    plt.ioff()
    plt.show()

class MetricPlotter(Callback):
  """
  Metric Plotter
  -----
    A callback that plots the  validation loss in real-time during training. Just place this premade callback in the compile method of the model.
    This callback assumes that validation_split is used during training.
  """
  def __init__(callbackself, metricindex=0, scale='log'):
    callbackself.metric_index = metricindex
    callbackself.scale = scale
    callbackself.fig = None
    callbackself.ax = None
    callbackself.lines = {}
      
  def initialization(callbackself, *args, **kwargs):
    plt.ion()
    callbackself.fig, callbackself.ax = plt.subplots()
    callbackself.ax.set_xlabel('Epoch')
    callbackself.ax.set_ylabel('Loss')
    callbackself.ax.set_yscale(callbackself.scale)
    callbackself.ax.grid(True)
    
    # Create line objects for training and validation loss
    callbackself.lines['metric'], = callbackself.ax.plot([], [])
      
  def after_epoch(callbackself, *args, **kwargs):
    
    x_data = list(range(kwargs.get('epoch', 0) + 1))
    y_data = [x[callbackself.metric_index] for x in kwargs.get('self', None).metrics_logs]
    
    callbackself.lines['metric'].set_data(x_data, y_data)
    
    # Dynamically adjust plot limits
    callbackself.ax.set_xlim(0, max(x_data) + 1 if x_data else 1)
    callbackself.ax.set_ylim(min(y_data) / 2, max(y_data) * 1.1)
    
    # Draw and flush events to update the plot
    callbackself.fig.canvas.draw()
    callbackself.fig.canvas.flush_events()
      
  def end(*args, **kwargs):
    plt.ioff()
    plt.show()

