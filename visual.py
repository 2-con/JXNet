"""
Visualizer
-----
Visualizes 2D/1D arrays through the terminal.
"""

import time
from tools.arraytools import shape
from tools.math import sgn
import numpy as np
import matplotlib.pyplot as plt
from tools.scaler import argmax

def image_display(input:list, **kwargs) -> None:
  """
  Black and White Display
  -----
    Displays the 2D/1D array as a monochromatic image
  -----
  Args
  -----
  - (Optional) boundary        (float) : the boundary seperating light from dark values
  - (Optional) upper_tolerance (float) : pixels with values higher than the boundary + upper_tolerance will be displayed as "■"
  - (Optional) lower_tolerance (float) : pixels with values lower than the boundary - lower_tolerance will be displayed as "□"
  - (Optional) title           (string) : the title of the image
  """
  upper_tolerance = kwargs.get('upper_tolerance', 0)
  lower_tolerance = kwargs.get('lower_tolerance', 0)
  boundary        = kwargs.get('boundary', 0)
  title = kwargs.get('title', '')
  
  print(title)
  
  if type(input[0]) == list: # probably a 2D array
    for row in input:
      for pixel in row:
        if pixel > boundary + upper_tolerance:
          print("■", end=" ")

        elif pixel < boundary - lower_tolerance:
          print("□", end=" ")

        else:
          print("•", end=" ")
      print()

  else: # probably a 1D array
    for pixel in input:
      if pixel > boundary + upper_tolerance:
        print("■", end=" ")

      elif pixel < boundary - lower_tolerance:
        print("□", end=" ")

      else:
        print("•", end=" ")
    print()

def array_display(data, pad:int = 2, start_indent:int = 0, decimals:int = 10, **kwargs) -> None:
  """
  Array Display
  -----
    Displays any n-dimensional array nicely
  -----
  Args
  -----
  - data                    (list or tuple) : the data to be shown
  - (optional) pad          (int)           : the number of spaces to indent the data
  - (optional) start_indent (int)           : the initial number of spaces to indent the data
  - (Optional) decimals     (string)        : displays how much decimals to show, assumeing its a float
  """
  data_shape = shape(data)
  indent = ' ' * start_indent
  # do not pass these as kwargs except during recursion
  maxnum = kwargs.get('maxnum', 0)
  padding = kwargs.get('padding', [])
  depth = kwargs.get('depth', 0)
  
  if type(data) in (int, float, str, bool):
    print(data)
    return
  
  # padding
  # multidimensional
  for img in data:
    if type(img) in (list, tuple) and len(shape(img)) == 2:
      
      maxnum = 0
      
      for point in img:
        for subpoint in point:
          
          if type(subpoint) in (float, int):
            
            maxnum = len(str(round(subpoint, decimals))) if len(str(round(subpoint, decimals))) > maxnum else maxnum
        
          else:
            maxnum = len(str(subpoint)) if len(str(subpoint)) > maxnum else maxnum
        
      padding.append(maxnum)
  
  #2d
  if type(data) in (list, tuple) and len(shape(data)) == 2 and depth==0:
      
    maxnum = 0
    
    for point in data:
      for subpoint in point:
        
        if type(subpoint) in (float, int):
          
          maxnum = len(str(round(subpoint, decimals))) if len(str(round(subpoint, decimals))) > maxnum else maxnum
      
        else:
          maxnum = len(str(subpoint)) if len(str(subpoint)) > maxnum else maxnum
      
    padding.append(maxnum)
  
  # 1d
  if type(data) in (list, tuple) and len(shape(data)) == 1 and depth==0:
    maxnum = 0
    padding = []
    for point in data:
        
      if type(point) in (float, int):
        
        maxnum = len(str(round(point, decimals))) if len(str(round(point, decimals))) > maxnum else maxnum
    
      else:
        maxnum = len(str(point)) if len(str(point)) > maxnum else maxnum
      
    padding.append(maxnum)
    
  # display
  if all(type(item) not in (list, tuple) for item in data):
    
    print(indent + "[ ", end='')
    maxnum = max(padding)
    
    for index, point in enumerate(data):
      
      if type(point) in (float, int):
        print(round(point, decimals), end=' ' + ' ' * ( maxnum - len(str(round(point, decimals)))) )
        
      elif type(point) == str:
        print(f"'{point}'", end=' ' + ' ' * ( maxnum - len(str(point))) )
        
      elif type(point) == bool:
        print(point, end=' ' + ' ' * ( maxnum - len(str(point))) )
        
      else:
        print(f"<{point.__name__}>", end=' ' + ' ' * ( maxnum - len(str(point))) )
        
    print("]")
    
  else:
    print(indent + '[')
    for i in range(data_shape[-1]):
      array_display(data[i], pad, start_indent=start_indent + pad, decimals=decimals, maxnum=maxnum, padding=padding, depth=depth + 1)
      
    print(indent + ']')

def dictionary_display(data: dict, pad: int = 2):
  """
  Dictionary Display
  -----
    Displays any dictionary nicely
  -----
  Args
  -----
  - data (dict) : the dictionary to be shown
  - (optional) pad (int) : the number of spaces to indent the dictionary
  """
  print('{') if pad == 2 else None
  
  for key, value in data.items():
    print(' ' * pad, end='')
    print(f"'{key}': ", end='')
    
    if type(value) == dict:
      print('{')
      dictionary_display(value, pad + 2)
      print(' ' * pad + '}')
      
    elif type(value) in (list, tuple):
      array_display(value, start_indent=pad + 2)
      
    else:
      print(repr(value), end='')
      
    print()
    
  if pad == 2:
    print('}')

def tree_display(start_node, pad:int = 2) -> None:
  """
  Tree Display
  -----
    Prints a tree structure in a hierarchical format. for this to function, the tree in question must be a PyNet Node object.
  -----
  Args
  -----
  - start_node (PyNet Node object) : The node object from which to start visualization.
  - indent_width (int) : The number of spaces to use for each level of indentation.
  """
  if start_node is None:
    print("Tree is empty or node is None.")
    return

  def _print_node_recursive(node, prefix=""):
    if node is None:
      return

    print(f"{prefix}■ Args: {node.args} ║ Kwargs: {node.kwargs}")
    
    # Recurse for children
    for i, child in enumerate(node.children):
      
      is_last_child = i == len(node.children) - 1
      new_prefix_branch = prefix + (" " + " " * (pad - 1) if not is_last_child else " " * pad)
      
      _print_node_recursive(child, new_prefix_branch)

  # Start the recursive printing from the given node
  _print_node_recursive(start_node, prefix="")

def char_display(string, speed, wrap = 50) -> None:
  """
  Charachter Display
  -----
    Displays the entire string charachter by charachter
  -----
  Args
  -----
  - string (string) : the string to display
  - speed  (float)  : how fast to print
  - wrap   (int)    : how far the text can go before wrapping
  """
  
  count = 0
  for i in string:
    count += 1
    time.sleep(speed)
    
    if count % wrap == 0:
      print()
    
    print(i, end='', flush=True)
  print()

def word_display(string, speed, wrap = 50) -> None:
  """
  Charachter Display
  -----
    Displays the entire string word by word
  -----
  Args
  -----
  - string (string) : the string to display
  - speed  (float)  : how fast to print
  - wrap   (int)    : how far the text can go before wrapping
  """
  count = 0
  for i in string.split():
    count += 1
    time.sleep(speed)
    
    if count % wrap == 0:
      print()
    
    print(i, end=' ', flush=True)
  print()

def display_boundary(model, features:list, targets:list, *args, **kwargs) -> None:
  """
  Plot Boundary
  -----
    Plots the decision boundary of a 2D binary classification model. The provided features must only contain two features (2D) for this function to work.
  -----
  Args
  -----
  - model (PyNet Model) : the model to construct the decision boundary from
  - features (list) : the features to plot
  - targets   (list) : the targets to plot, helps color-in the points
  - (Optional) title (str) : the title of the plot
  - (Optional) zoom (int or float) : the initial zoon of the plot
  - (Optional) cmap (matplotlib.colors.Colormap) : the colormap to use for the decision regions
  - (Optional) transparency (float) : the transparency of the decision regions
  - (Optional) n_points (int) : the number of points to use for the meshgrid along each dimension (quality of the boundary)
  """
  
  cmap = kwargs.get('cmap', plt.cm.RdBu)
  title = kwargs.get('title', '')
  alpha = kwargs.get('transparency', 0.5)
  n_points = kwargs.get('n_points', 300)
  zoom = kwargs.get('zoom', 1)
  
  ax=None
  X_np = np.asarray(features)
  y_np = np.asarray(targets)
  
  if X_np.shape[1] != 2:
    raise ValueError("X must be a 2D array to plot a decision boundary.")

  # plotting
  
  if ax is None:
    fig, ax = plt.subplots(figsize=(7, 7))

  x_min, x_max = X_np[:, 0].min() - zoom, X_np[:, 0].max() + zoom
  y_min, y_max = X_np[:, 1].min() - zoom, X_np[:, 1].max() + zoom

  xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_points),
                       np.linspace(y_min, y_max, n_points))

  # prepare meshgrid for prediction
  meshgrid_points = np.c_[xx.ravel(), yy.ravel()]
  predictions_input_for_model = meshgrid_points.tolist()
  
  if hasattr(model, 'push'):
    Z_list_output = [argmax(a) for a in [model.push(x) for x in predictions_input_for_model]]
  else:
    Z_list_output = [model.predict(x) for x in predictions_input_for_model]

  # convert back to a np array for reshaping
  Z = np.asarray(Z_list_output).reshape(xx.shape)

  # plot regions and data points
  ax.contourf(xx, yy, Z, cmap=cmap, alpha=alpha)
  
  if hasattr(model, 'push'):
    scatter = ax.scatter(X_np[:, 0], X_np[:, 1], c=[ argmax(x) for x in targets], cmap=cmap, edgecolor='k', s=80, zorder=2)
  else:
    scatter = ax.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap=cmap, edgecolor='k', s=80, zorder=2)

  ax.set_title(title)
  ax.set_xlim(xx.min(), xx.max())
  ax.set_ylim(yy.min(), yy.max())

  plt.show()
  

