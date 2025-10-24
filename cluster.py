"""
Cluster
=====
  Unsupervised learning algorithms for clustering data points.
-----
Provides
-----
- KMeans
- DBScan
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
from tools.arraytools import transpose, distance

class KMeans:
  def __init__(self):
    """
    K-Means clustering algorithm, works by partitioning the data into k clusters.
    """
    self.centroids = []
    self.is_fitted = False
  
  def compile(self, maximum_iterations:int, centroids:int, distance_metric:int, *args, **kwargs):
    """
    Compile
    -----
      Compiles the model to be ready for training.
    -----
    Args
    -----
    - maximum_iterations (int) : Decides the maximum amount of iterations allowed
    - centroids          (int) : Decides the amount of centroids (classes) allowed; Initialized randomly.
    - distance_metric    (int) : Controls the distance metric used. 1 for L1 (manhattan), 2 for L2 (euclidian)
    """
    self.is_compiled = True
    self.centroids_amount = centroids
    self.maximum_iterations = maximum_iterations
    self.ldim = distance_metric
    
    if maximum_iterations <= 0:
      raise ValueError("Epochs must be greater than 0")
    if centroids <= 0:
      raise ValueError("Number of centroids must be greater than 0")
    
  def fit(self, data, **kwargs):
    """
    Fit
    -----
      Fits the model to the data, fitting the centeroids to the data.
    -----
    Args:
    -----
    - features (list) : the dataset, must be a 2D array
    """
    self.data = data
    
    def nearest_centroid(point:list):
      return min(range(len(self.centroids)), key=lambda x: distance(point, self.centroids[x],self.ldim))
    
    def average(data:list):
      return [sum(feature_row)/len(data) for feature_row in transpose(data)]
    
    self.centroids = [(random.normalvariate(avg,1) for avg in average(data)) for _ in range(self.centroids_amount)]
    
    for _ in range(self.maximum_iterations):
      static_centroids = 0
      for centroid_index in range(len(self.centroids)):
        valid_points = []
        
        for point in data:
          if nearest_centroid(point) == centroid_index:
            valid_points.append(point)
        
        if self.centroids[centroid_index] == average(valid_points):
          static_centroids += 1
          
        self.centroids[centroid_index] = average(valid_points)
      
      if static_centroids == len(self.centroids):
        break
      
  def classify(self, return_format:str):
    """
    classify
    -----
      Return the predicted clusters
    -----
    Args
    -----
    - return_format (str) : decides the format, either 'labeled' where the labels are provided or 'centroid' where the centroids are provided
    """
    
    def nearest_centroid(point:list):
      return min(range(len(self.centroids)), key=lambda x: distance(point, self.centroids[x], self.ldim))
    
    if return_format == "labeled":
      return [nearest_centroid(point) for point in self.data]
    
    elif return_format == "centroid":
      return [centroid for centroid in self.centroids]
    
    else:
      raise ValueError(f"Unknown return format '{return_format}'")

class Kmedoid:
  def __init__(self):
    """
    K-Means clustering algorithm, works by partitioning the data into k clusters.
    """
    self.medoids = []
    self.is_fitted = False
  
  def compile(self, maximum_iterations:int, medoids:int, distance_metric:int, *args, **kwargs):
    """
    Compile
    -----
      Compiles the model to be ready for training.
    -----
    Args
    -----
    - maximum_iterations (int) : Decides the maximum amount of iterations allowed
    - centroids          (int) : Decides the amount of centroids (classes) allowed; Initialized randomly.
    - distance_metric    (int) : Controls the distance metric used. 1 for L1 (manhattan), 2 for L2 (euclidian)
    """
    self.is_compiled = True
    self.centroids_amount = medoids
    self.maximum_iterations = maximum_iterations
    self.ldim = distance_metric
    
    if maximum_iterations <= 0:
      raise ValueError("Epochs must be greater than 0")
    if medoids <= 0:
      raise ValueError("Number of medoids must be greater than 0")
    
  def fit(self, data, **kwargs):
    """
    Fit
    -----
      Fits the model to the data, fitting the centeroids to the data.
    -----
    Args:
    -----
    - features (list) : the dataset, must be a 2D array
    """
    self.data = data
    
    def nearest_centroid(point:list):
      return min(range(len(self.medoids)), key=lambda x: distance(point, self.medoids[x], self.ldim))
    
    def nearest_point(point:list):
      return min(range(len(self.data)), key=lambda x: distance(point, self.data[x], self.ldim))
    
    def average(data:list):
      return [sum(feature_row)/len(data) for feature_row in transpose(data)]
    
    self.medoids = [data[nearest_point(( random.normalvariate(avg,1) for avg in average(data)))] for _ in range(self.centroids_amount)]
    
    for _ in range(self.maximum_iterations):
      static_centroids = 0
      for centroid_index in range(len(self.medoids)):
        valid_points = []
        
        for point in data:
          if nearest_centroid(point) == centroid_index:
            valid_points.append(point)
        
        if self.medoids[centroid_index] == data[nearest_point(average(valid_points))]:
          static_centroids += 1
          
        self.medoids[centroid_index] = data[nearest_point(average(valid_points))]
      
      if static_centroids == len(self.medoids):
        break
      
  def classify(self, return_format:str):
    """
    classify
    -----
      Return the predicted clusters
    -----
    Args
    -----
    - return_format (str) : decides the format, either 'labeled' where the labels are provided or 'centroid' where the centroids are provided
    """
    def nearest_medoids(point:list):
      return min(range(len(self.medoids)), key=lambda x: distance(point, self.medoids[x], self.ldim))
    
    if return_format == "labeled":
      return [nearest_medoids(point) for point in self.data]
    
    elif return_format == "centroid":
      return [centroid for centroid in self.medoids]
    
    else:
      raise ValueError(f"Unknown return format '{return_format}'")

class DBScan:
  def __init__(self):
    """
    DBScan clustering algorithm
    """
    self.is_fitted = False
  
  def compile(self, epsilon:float, min_points:int, distance_metric:int, *args, **kwargs):
    """
    Compile
    -----
      Compiles the model to be ready for training.
    -----
    Args
    -----
    - epsilon         (float) : The distance of a point to be considered a neighbor
    - min_points      (int)   : How many neighbors are required to turn into a core point
    - distance_metric (int)   : Controls the distance metric used. 1 for L1 (manhattan), 2 for L2 (euclidian)
    """
    self.is_compiled = True
    self.epsilon = epsilon
    self.ldim = distance_metric
    self.min_points = min_points
  
  def fit(self, data, **kwargs):
    """
    Fit
    -----
      Fits the model to the data, fitting the centeroids to the data.
    -----
    Args:
    -----
    - features (list) : the dataset, must be a 2D array
    """
    
    self.data = data
    
    def amount_of_neighbors(point, data):
      ans = 0
      for pt in data:
        if distance(point, pt, self.ldim) == 0:
          pass
        elif distance(point, pt, self.ldim) <= self.epsilon:
          ans += 1
      
      return ans
    
    def get_neighbors(point, data):
      ans = []
      
      for pt in data:
        if distance(point, pt, self.ldim) <= self.epsilon:
          ans.append(pt)
      
      return ans
    
    core_points = []
    for point in data:
      if amount_of_neighbors(point, data) >= self.min_points - 1:
        core_points.append(point)
    
    border_points = [pt for pt in data if pt not in core_points]
    
    self.classes = []
    while len(core_points) > 0:
      new_class = [core_points[0]]
      core_points.remove(core_points[0])
      
      # convert core points into the new class
      spreading = True
      while spreading:
        
        neighborless = 0
        for class_point in new_class:
          neighbors = get_neighbors(class_point, core_points)
          
          for neighbor in neighbors:
            if neighbor not in new_class:
              new_class.append(neighbor)
              core_points.remove(neighbor)
          
          if amount_of_neighbors(class_point, core_points) == 0:
            neighborless += 1
          
        if neighborless == len(new_class):
          spreading = False
      
      # convert border points into the new class
      for class_point in new_class:
        for neighbor in get_neighbors(class_point, data):
          if neighbor in border_points:
            border_points.remove(neighbor)
            new_class.append(neighbor)
    
      self.classes.append(new_class)
    
    # noise
    self.classes.append(border_points)
    
  def classify(self):
    """
    classify
    -----
      Return the predicted clusters
    """
    ans = []
    for point in self.data:
      for cluster_index in range(len(self.classes)):
        if point in self.classes[cluster_index]:
          ans.append(cluster_index)
      
    return ans

