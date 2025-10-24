"""
Data Field
=====
  Basic frameworks for data containers, some PyNet processes depends and inherits from these data containers to operate.
  but besides being foundational, these classes are also useful for more complex data structures.
"""

class Datacontainer:
  def __init__(self, data, *args, **kwargs):
    """
    PyNet Datacontainer
    -----
      A Base class for data containers. Its not meant to be used directly, more so to be inherited from and expanded on.
    -----
    Args
    -----
    - data : the data to be stored
    - args : any additional data
    - kwargs : any additional data
    """
    self.data = data
    self.args = args
    self.kwargs = kwargs
    
  def contents(self) -> None:
    print("Data")
    print(f"  {self.data}")
    print("Args")
    print(f"  {self.args}")
    print("KW Args")
    print(f"  {self.kwargs}")

class Node:
  def __init__(self, *args, **kwargs):
    """
    PyNet Node
    -----
      A general node class for data structures. can be used to create trees or linked lists.
    -----
    Args
    -----
    - (Optional) args (tuple)  : positional arguments
    - (Optional) kwargs (dict) : keyword arguments
    """
    
    self.args = args
    self.kwargs = kwargs
    
    self.parent = None
    self.children = []

  def add_child(self, child_node):
    """
    Add Child
    -----
      Adds a child node to the current node.
    -----
    Args
    -----
    - child_node (Node) : the child node to be added
    """
    
    if not isinstance(child_node, Node):
      raise TypeError("Child must be an instance of Node")
    
    self.children.append(child_node)
    child_node.parent = self
  
  def remove_child(self, child_node):
    """
    Remove Child
    -----
      Removes a child node from the current node.
    -----
    Args
    -----
    - child_node (Node) : the child node to be removed
    """
    
    if not isinstance(child_node, Node):
      raise TypeError("Child must be an instance of Node")
    
    if child_node in self.children:
      self.children.remove(child_node)
      child_node.parent = None
    else:
      raise ValueError("Child node not found in children")
  
  def get_child(self, index=None):
    """
    Get Child
    -----
      Returns all of the children of the current node. If an index is provided, returns the child at that index provided it has a child at the index.
    -----
    Args
    -----
    - index (int, optional) : if provided, returns the child at that index, otherwise returns all children
    -----
    Returns
    -----
    - list or Node : list of children nodes or a single child node if index is provided
    """
    
    if index is not None:
      if 0 <= index < len(self.children):
        return self.children[index]
      else:
        raise IndexError("Index out of range")
    
    return self.children
  
  def depth(self) -> int:
    """
    Depth
    -----
      Returns the level of the node.
    -----
    Returns
    -----
    - int : the level of the node
    """
    
    level = 1
    current_node = self
    while current_node.parent is not None:
      level += 1
      current_node = current_node.parent
    
    return level

  def get_first(self) -> "Node":
    """
    Get First
    -----
      Traverses upwards until a dead end is met before returning the node.
    -----
    Returns
    -----
    - Node : the first node in the tree
    """
    
    current_node = self
    while current_node.parent is not None:
      current_node = current_node.parent
    
    return current_node
  
  def get_last(self) -> list:
    """
    Get Last
    -----
      Traverses downwards until dead end(s) are encountered before returning the node(s).
    -----
    Returns
    -----
    - list : a list of all the last nodes
    """
    leaves = []
    stack = [self] # Start DFS from the current node

    while stack:
      
      node = stack.pop()
      
      # checks if the node is a leaf
      if len(node.children) == 0:
        leaves.append(node)
        
      else:
        for child in reversed(node.children):
          stack.append(child)
          
    return leaves

