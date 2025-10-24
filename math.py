"""
Math
-----
Contains useful math functions that are not part of Numpy or Python
"""

import math

def sgn(x):
  """
  Sign
  -----
    Returns the sign of a number. it eitehr returns 1 if the number is positive, -1 if the number is negative, or 0 if the number is 0.
  -----
  Args:
    x (int / float): The number to check the sign of.
  -----
  Returns:
    int: The sign of the number.
  """
  if x > 0:
    return 1
  elif x < 0:
    return -1
  else:
    return 0

def clamp(x, a, b):
  """
  Clamp
  -----
    Clamps a value within a specified range.
  -----
  Args:
    a (int or float): The minimum value of the range.
    b (int or float): The maximum value of the range.
    x (int or float): The value to clamp.
  """
  return max(min(x, b), a)

def lerp(a, b, t):
  """
  Linear Interpolation
  -----
    Linear interpolation between two values.
  -----
  Args:
    a (int or float): The start value.
    b (int or float): The end value.
    t (float): The interpolation factor, typically between 0 and 1.  
  """
  return a + t * (b - a)

def smoothstep(edge0, edge1, x):
  """Smooth Hermite interpolation between 0 and 1."""
  if x <= edge0:
    return 0
  if x >= edge1:
    return 1
  t = (x - edge0) / (edge1 - edge0)
  return t * t * (3 - 2 * t)

def ackermann(x,y):
  """
  Ackermann
  -----
    Computes the Ackermann function. this is not a practical tool but rather, an easter egg.
    be aware that this function notorious for its hyper exponential growth.
  -----
  Args:
    x (int): The first parameter of the Ackermann function.
    y (int): The second parameter of the Ackermann function.
  -----
  Returns:
    int: The result of the Ackermann function.
  """
  if x <= 0:
    return y + 1
  elif y <= 0:
    return ackermann(x-1, 1)
  else:
    return ackermann(x-1, ackermann(x, y-1))

def Variance(items:list):
  if not items:
    raise ValueError("elements cannot be empty")
  
  total = len(items)
  if total == 0:
    return 0
  
  variance = 0.0
  mean = sum(items) / total
  for item in items:
    variance += (item - mean) ** 2
  
  return variance

def majority(items:list):
  """
  Vote
  -----
    Returns the most common item in a list.
  -----
  Args:
    items (list): The list of items to vote on.
  -----
  Returns:
    The most common item in the list.
  """
  return max(set(items), key=items.count)