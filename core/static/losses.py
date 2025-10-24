import math

def Gini_impurity(items:list):
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

