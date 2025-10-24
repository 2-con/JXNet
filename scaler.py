"""
Scaler
-----
Scalers to normalize data or lists.
"""

import math
import statistics
import numpy as np

def standard_scaler(x:list):
  """
  Transform numerical data to have a mean of zero and a standard deviation of one
  """
  answer = []
  average = sum(x) / len(x)
  standard_deviation = float(np.std(x))

  for i in x:
    answer.append(((i - average) / standard_deviation) if standard_deviation != 0 else 0)
  return answer

def min_max_scaler(x:list, min:float, max:float):
  """
  Normalize all values from min to max
  """
  answer = []

  for i in x:
    answer.append((i - min) / (max - min))
  return answer

def max_abs_scaler(x:list):
  """
  Normalize all values with respect to the maximum absolute value
  """
  answer = []
  max_abs = max(abs(i) for i in x)

  for i in x:
    answer.append(i / (max_abs + 1e-8))
  return answer

def robust_scaler(x:list):
  """
  Scales the data based on the median and the interquartile range
  """
  answer = []
  q1 = statistics.median_low(x)
  q3 = statistics.median_high(x)
  iqr = q3 - q1

  for i in x:
    answer.append(((i - q1) / iqr) if iqr != 0 else 0)
  return answer

def softmax(x:list):
  exp_x = [math.exp(i-max(x)) for i in x]
  return [(i-max(x)) / sum(exp_x) for i in exp_x]

def argmax(x):
  return x.index(max(x))

def argmin(x):
  return x.index(min(x))