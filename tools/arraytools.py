"""
Array Tools
-----
  Array tools are tools that helps with working with python arrays (lists). JXNet will always vectorize all operations without ever needing tools from this file
  except for non-vectorized models.
  
Provides:
-----
- transpose
- distance
"""

def transpose(input):
  """
  Transpose
  -----
    Transposes a 2D array (rows become columns, columns become rows)
  -----
  Args
  -----
  input (2D array) : the array to transpose
  
  Returns
  -----
    2D array
  """
  answer = []
  for i in range(len(input[0])):
    row = []
    for j in range(len(input)):
      row.append(input[j][i])
    answer.append(row)

  return answer

def distance(a, b, l:int):
  """
  Distance
  -----
    Returns the distance between two points in a l-dimensional space. 
  -----
  Args
  -----
  a (list) : the first point
  b (list) : the second point
  l (int)  : the dimension of the space
  """
  if len(a) != len(b):
    raise ValueError("a and b must be the same length")
  
  if l <= 0:
    raise ValueError("L must be positive")
  
  return sum((x-y)**l for x,y in zip(a,b))**(1/l)