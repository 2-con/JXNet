"""
Array Tools
-----
A collection of tools for working with arrays, mainly 1D and 2D arrays
"""
  
import random
import typing

def correlate(matrix, kernel):
  """
  Convolve
  -----
    Convolve a 2D matrix with a 2D kernel.
  -----
  Args
  -----
    matrix (list of list of int): The input matrix.
    kernel (list of list of int): The kernel to convolve with.
  
  Returns
  -----
    list of list of int: The convolved matrix.
  """
  m_rows = len(matrix)
  m_cols = len(matrix[0])
  k_rows = len(kernel)
  k_cols = len(kernel[0])
  
  # Calculate the dimensions of the output matrix
  output_rows = m_rows - k_rows + 1
  output_cols = m_cols - k_cols + 1
  
  # Initialize the output matrix with zeros
  output = [[0] * output_cols for _ in range(output_rows)]
  
  # Perform convolution
  for i in range(output_rows):
    for j in range(output_cols):
      for ki in range(k_rows):
        for kj in range(k_cols):
          output[i][j] += matrix[i + ki][j + kj] * kernel[ki][kj]
  
  return output

def reshape(input_data, new_shape):
  """
  Reshape
  -----
    Reshape an array into a new shape, input array can be any shape and new shape can be any shape as long as 
    there is enough elements to fill the new shape.
  -----
  Args
  -----
  input_data (list)          : The input array to be reshaped.
  new_shape  (list or tuple) : The desired shape for the reshaped array.
  
  Returns
  -----
  list: The reshaped array.
  """

  # --- Main reshape logic starts here ---
  pile = flatten(input_data)
  total_elements = len(pile)

  # Convert new_shape to a list for mutable processing
  new_shape_list = list(new_shape)[::-1]
  
  # Handle negative dimension inference
  num_neg_one = new_shape_list.count(-1)
  if num_neg_one > 1:
    raise ValueError("Only one dimension can be inferred (-1)")
  
  elif num_neg_one == 1:
    
    inferred_dim_index = new_shape_list.index(-1)
    product_of_known_dims = 1
    
    for i, dim in enumerate(new_shape_list):
      if i != inferred_dim_index:
        if dim <= 0: # Ensure known dimensions are positive
          raise ValueError(f"Cannot infer dimension at index {inferred_dim_index} if another dimension is zero or negative: {new_shape}")
        
        product_of_known_dims *= dim
    
    if product_of_known_dims == 0:
      
      # If product is 0, and total_elements is not 0, it's impossible to infer
      if total_elements != 0:
        raise ValueError(f"Cannot infer dimension when product of other dimensions is zero, and input is not empty")
      
      inferred_value = 0 # If total elements is 0, the inferred dim must be 0
    
    elif total_elements % product_of_known_dims != 0:
      raise ValueError(f"Cannot reshape array of size {total_elements} into shape {new_shape}")
    
    else:
      inferred_value = total_elements // product_of_known_dims
    
    new_shape_list[inferred_dim_index] = inferred_value

  # Validate final shape (all dimensions must be positive)
  for dim in new_shape_list:
    if dim < 0:
      raise ValueError(f"Invalid dimension value: {dim} in {new_shape}. Only -1 is allowed for inference")
          
  # Calculate expected total elements for the new shape
  expected_total_elements = 1
  for dim in new_shape_list:
    
    expected_total_elements *= dim

  # Check for size mismatch
  if expected_total_elements != total_elements:
    raise ValueError(f"Cannot reshape array of size {total_elements} into shape {tuple(new_shape_list)} (requires {expected_total_elements} elements)")

  # Helper function 3: Recursively build the multidimensional array
  def build(dim_index, current_pile_index):
    
    if dim_index == len(new_shape_list):
      
      if len(new_shape_list) == 0: # Reshaping to scalar ()
        
        if current_pile_index >= total_elements: # Should not happen if size check is correct
          raise ValueError("Internal error: Pile exhausted when forming scalar.")
        
        return pile[current_pile_index], current_pile_index + 1
      
      raise RuntimeError("Unexpected base case in recursive array builder.")
        
    elif dim_index == len(new_shape_list) - 1:
      
      # Base case: Forming the innermost dimension (e.g., a row in a 2D array, or a vector in a 3D array)
      size_of_last_dim = new_shape_list[dim_index]
      segment = pile[current_pile_index : current_pile_index + size_of_last_dim]
      return segment, current_pile_index + size_of_last_dim
    
    else:
      # Recursive step: Forming a higher-level dimension (a list of sub-arrays)
      num_sub_arrays = new_shape_list[dim_index]
      current_dimension_list = []
      
      for _ in range(num_sub_arrays):
        
        # Recursively build the next lower dimension
        sub_array, current_pile_index = build(dim_index + 1, current_pile_index)
        current_dimension_list.append(sub_array)
      
      return current_dimension_list, current_pile_index

  # --- Start the recursive array construction ---
  if len(new_shape_list) == 0: # Target shape is () for a scalar output
    if total_elements != 1:
      raise ValueError(f"Cannot reshape array of size {total_elements} into scalar (requires 1 element)")
    return pile[0]
      
  final_reshaped_array, final_pile_index = build(0, 0)
  
  # Final check (should ideally be covered by initial total_elements check)
  if final_pile_index != total_elements:
    raise RuntimeError("Internal error: Mismatch in elements consumed during recursive reshaping")

  return final_reshaped_array
  
def flatten(data):
  """
  Flatten
  -----
    Flattens any 2D or 3D array into a 1D array
  -----
  Args
  -----
  input (2D or 3D array) : the array to flatten
  
  Returns
  -----
    1D List
  """
  flattened_list = []
  if isinstance(data, list):
    for item in data:
      flattened_list.extend(flatten(item))
  else:
    flattened_list.append(data)
  return flattened_list

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

def concat(*input):
  """
  Concatenate
  -----
    Concatenates any number of 1D/2D arrays horizontally.
  -----
  Args
  -----
    input (2D array) : the array to horizontally concatenate
  """
  answer = []
  
  if len(shape(input[0])) > 1:
    
    for index, image in enumerate(input):
      
      if shape(image) != shape(input[0]):
        raise ValueError(f"All tensors must be the same shape, tensor {index} has shape {shape(image)} while tensor 0 has shape {shape(input[0])}")
      
      answer += transpose(image)
    
    return transpose(answer)

  else:
    
    for vect in input:
      answer += vect
      
    return answer

def total(*args):
  """
  Pointwise summation
  -----
    Pointwise summation of any number of 1D/2D arrays
  -----
  Args
  -----
    input (2D array) : the array to transpose
  """
  arrays = list(args)
  
  if not arrays:
    raise ValueError("Input list of arrays cannot be empty.")

  def _get_shape(arr):
    if arr is None:
      return (0,)
    if not isinstance(arr, (list, tuple)):
      return () # Scalar
    if not arr:
      return (0,) # Empty list/tuple at this level
    
    current_dim_size = len(arr)
    sub_shape = _get_shape(arr[0])
    return (current_dim_size,) + sub_shape

  # 1. Get the shape of the first array and validate all input arrays
  #    have the same shape.
  first_array_shape = _get_shape(arrays[0])

  if not first_array_shape: # Check if the first 'array' is actually a scalar or empty
    raise ValueError("Input arrays must be 1D or 2D lists/tuples, not scalars or empty.")
  
  if len(first_array_shape) > 2:
      raise ValueError("This function is designed for 1D or 2D arrays only.")
  
  # Initialize the result array with zeros (or the first array's values)
  # based on the determined shape.
  if len(first_array_shape) == 1: # 1D array
    result = [0] * first_array_shape[0]
  elif len(first_array_shape) == 2: # 2D array
    rows, cols = first_array_shape
    result = [[0 for _ in range(cols)] for _ in range(rows)]
  else:
    # just in case there is more funny shapes.
    raise ValueError("Unexpected array dimension. Only 1D and 2D supported.")

  # 2. Perform pointwise addition
  for i, arr in enumerate(arrays):
    current_array_shape = _get_shape(arr)
    if current_array_shape != first_array_shape:
      raise ValueError(f"Array at index {i} has shape {current_array_shape}, "
                       f"but expected shape {first_array_shape}. All arrays must have the same shape.")

    if len(first_array_shape) == 1: # 1D addition
      for j in range(first_array_shape[0]):
        result[j] += arr[j]
    elif len(first_array_shape) == 2: # 2D addition
      rows, cols = first_array_shape
      for r in range(rows):
        for c in range(cols):
          result[r][c] += arr[r][c]

  return result

def generate_array(*args, **kwargs):
  """
  Generate array
  -----
    generate an array of specified dimensions
  -----
  Args
  -----
  args (int) : the dimensions of the array
  value (int) : the default value of the array
  
  Returns
  -----
    List / Array
  """

  value = kwargs.get('value', 0)

  answer = [value for _ in range(args[0])]

  for dimention in args[1:]:
    if dimention < 0:
      raise ValueError("Dimensions cannot be negative")

    answer = [answer[:] for _ in range(dimention)]

  return answer

def generate_random_array(*args, **kwargs):
  """
  Genrate random array
  -----
    Creates an array of specified dimensions with random values
  -----
  Args
  -----
  *args (int) : the dimensions of the array
  min  (int) : the minimum value of the array
  max  (int) : the maximum value of the array
  
  Returns
  -----
    List / Array
  """

  min = kwargs.get('min', 0)
  max = kwargs.get('max', 1)

  if not all(isinstance(dim, int) for dim in args[::-1]):
    raise TypeError("All dimensions must be integers.")
  if not all(dim > 0 for dim in args[::-1]):
    raise ValueError("All dimensions must be greater than 0.")
  if min > max:
    raise ValueError("min_val must be less than or equal to max_val.")

  def _generate_inner(dimensions):
    """
    Recursive helper function to generate the nested lists.
    """
    if not dimensions:
      return random.uniform(min, max)  # Base case: 0 dimensions
    else:
      return [_generate_inner(dimensions[1:]) for _ in range(dimensions[0])]

  return _generate_inner(args[::-1])

def shape(input):
  """
  Shape
  -----
    Returns the shape of an array
  -----
  Args
  -----
  input (ndim array) : the array to count
  """
  def recursive(input_tensor):
    if input_tensor is None:
      return (0,)

    if not isinstance(input_tensor, (list, tuple)):
      return ()

    if not input_tensor:
      return (0,) # An empty dimension

    # Get the size of the current dimension
    current_dimension_size = len(input_tensor)

    sub_shape = shape(input_tensor[0])

    # Combine the current dimension's size with the sub-shape
    return sub_shape + (current_dimension_size,)

  if input == None:
    return None

  return recursive(input)

def mirror(input, axis:str):
  """
  Mirror
  -----
    Mirrors a 2D array along the X or Y axis
  -----
  Args
  -----
  input (2D array) : the array to mirror
  axis (string) : the axis to mirror the array along 'X' or 'Y'
  
  Returns
  -----
    2D array
  """
  if axis == 'Y':
    return [row[::-1] for row in input]

  elif axis == 'X':
    return input[::-1]

  else:
    raise ValueError("axis must be 'X' or 'Y'")

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