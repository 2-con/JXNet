import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt

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

class RealTimePlotter(Callback):
  """
  RealTimePlotter
  -----
    A callback that plots the training and validation loss in real-time during training. Just place this premade callback in the compile method of the model.
    This callback assumes that validation_split is used during training.
  """
  def __init__(callbackself):
    callbackself.fig = None
    callbackself.ax = None
    callbackself.lines = {}
      
  def initialization(callbackself, *args, **kwargs):
    plt.ion()  # Turn on interactive mode
    callbackself.fig, callbackself.ax = plt.subplots()
    callbackself.ax.set_xlabel('Epoch')
    callbackself.ax.set_ylabel('Loss')
    callbackself.ax.set_yscale('log')
    callbackself.ax.grid(True)
    
    # Create line objects for training and validation loss
    callbackself.lines['train'], = callbackself.ax.plot([], [])
    callbackself.lines['validation'], = callbackself.ax.plot([], [])
      
  def after_epoch(callbackself, *args, **kwargs):
        
    x_data = list(range(kwargs.get('epoch', 0) + 1))
    y_data_train = kwargs.get('self', None).error_logs
    y_data_validation = kwargs.get('self', None).validation_error_logs
    
    callbackself.lines['train'].set_data(x_data, y_data_train)
    callbackself.lines['validation'].set_data(x_data, y_data_validation)
    
    # Dynamically adjust plot limits
    callbackself.ax.set_xlim(0, max(x_data) + 1 if x_data else 1)
    
    callbackself.ax.set_ylim(min(min(y_data_train), min(y_data_validation)) - 0.1, max(max(y_data_train), max(y_data_validation)) + 0.1)
    
    # Draw and flush events to update the plot
    callbackself.fig.canvas.draw()
    callbackself.fig.canvas.flush_events()
      
  def end(*args, **kwargs):
    plt.ioff()
    plt.show()
