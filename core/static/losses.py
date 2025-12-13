"""
Losses
=====
  Loss functions are ways to measure the error of a model from its desired output. Loss functions here are for algorithms that does not use gradient
  or backpropagation methods to optimize their system.

Provides:
- Gini impurity
- Entropy
"""

import math

def Gini_impurity(items:list):
  """
  Gini Impurity
  -----
    Gini impurities are defined as
    1 - the sum of the squared probabilities of each class. For this implimentation, any different datatype will be registered
    as a new class.
  """
  if not items:
    return 0
  
  total = len(items)
  if total == 0:
    return 0
  
  class_counts = {}
  
  for label in items:
    if label not in class_counts:
      class_counts[label] = 1
    else:
      class_counts[label] += 1
  
  gini = 1.0
  for count in class_counts.values():
    probability = count / total
    gini -= probability ** 2
  
  return gini

def Entropy(items:list):
  """
  Entropy
  -----
    Entropy is defined as the negative sum of the probabilities of each class. For this implimentation, any different datatype will be registered
    as a new class.
  """
  if not items:
    return 0
  
  total = len(items)
  if total == 0:
    return 0
  
  class_counts = {}
  
  for label in items:
    if label not in class_counts:
      class_counts[label] = 1
    else:
      class_counts[label] += 1
  
  entropy = 0.0
  for count in class_counts.values():
    probability = count / total
    if probability > 0:
      entropy -= probability * math.log(probability, 2)
  
  return entropy

