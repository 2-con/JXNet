"""
Classifier
=====
  Classification algorithms for PyNet, even though some of these algorithms are can be used for regression,
  any label is considered a class and will be treated as such.
-----
Provides
-----
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Naive Bayes (Gaussian, Multinomial and Bernoulli models supported)
- SVM (Support Vector Machine)
- MSVM (Multiclass Support Vector Machine)
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.arraytools import shape, transpose, distance
from core.static.losses import Entropy, Gini_impurity
from core.static.datafields import Node as n
from core.static.datafields import Datacontainer as dc
import random
import math

class Key:
  METRICS = {
    'accuracy': lambda y_true, y_pred: sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true),
    'precision': lambda y_true, y_pred: sum(1 for true, pred in zip(y_true, y_pred) if true == pred and pred == 1) 
                                      / sum(1 for pred in y_pred if pred == 1),
    'recall': lambda y_true, y_pred: sum(1 for true, pred in zip(y_true, y_pred) if true == pred and pred == 1) 
                                   / sum(1 for true in y_true if true == 1),
    'f1_score': lambda y_true, y_pred: 
      (2 * (sum(1 for true, pred in zip(y_true, y_pred) if true == pred and pred == 1) / sum(1 for pred in y_pred if pred == 1)) ** 2) 
        / ((sum(1 for true, pred in zip(y_true, y_pred) if true == pred and pred == 1) / sum(1 for pred in y_pred if pred == 1)) * 2) 
        if (sum(1 for true, pred in zip(y_true, y_pred) if true == pred and pred == 1) > 0) else 0.0,
    'roc_auc': lambda y_true, y_pred: 
     sum(1 for true, pred in zip(y_true, y_pred) if true == pred and pred == 1) / 
    (sum(1 for true in y_true if true == 1) + sum(1 for true in y_true if true == 0)) # simplified version
  }

class KNN:
  def __init__(self):
    """
    K-Nearest Neighbors
    -----
      Predicts the class of a point based on its k-nearest neighbors
    """
    self.is_compiled = False
    
  def compile(self, neighbors:int, distance_metric:int):
    """
    Compile
    -----
      Compiles the model to be ready for training, setting the number of neighbors to consider
    -----
    Args
    -----
    - neighbors       (int) : the number of neighbors to consider before classifying
    - distance_metric (int) : the L-Dimention measurement to use. 1 for L1 (manhattan), 2 for L2 (euclidian)
    """
    self.is_compiled = True
    self.neighbors = neighbors
    self.ldim = distance_metric
    
  def fit(self, features, labels):
    """
    Fit
    -----
      Fits the model to the training data. But since KNN does not have a training phase, this method stores the training data
    -----
    Args
    -----
    - features (list) : the features of the dataset, must be a 2D array
    - labels   (list) : the labels of the dataset, must be a 2D array
    """
    self.features = features
    self.labels = labels
    
  def predict(self, point):
    """
    Predict
    -----
      Predicts the class of a point, the model must be compiled before using this method.
    -----
    Args:
    -----
    - point (list) : the point to predict, a vector (list of intigers or floats)
    """
    
    if not self.is_compiled:
      raise SystemError("Model must be compiled before predicting anything")

    distances = []
    
    for feature in self.features:
      if len(feature) != len(point): # ensure feature and point are the same length
        raise SystemError(f"Feature and point must be the same length, {feature} ({len(feature)}) is incompatible with {point} ({len(point)})")
      
      elif len(shape(point)) != 1: # ensures feature and point are vectors (multidimentional points)
        raise SystemError(f"point must be a vector (list of intigers or floats)")
      
      else: # calculate the L1 distance between the point and the feature
        
        distances.append( distance(point, feature, self.ldim) )
        
    # sort the distances and get the indices of the k-nearest neighbors
    indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.neighbors]
    
    # get the labels of the k-nearest neighbors
    neighbors_labels = [self.labels[i] for i in indices]
    
    # return the most common label
    return max(set(neighbors_labels), key=neighbors_labels.count)
  
  def evaluate(self, features, labels, metrics:list):
    """
    Evaluate
    -----
      Evaluates the model on a test set using a specified metric.
    -----
    Args
    -----
    - features (list) : the features of the dataset, must be a 2D array
    - labels   (list) : the labels of the dataset, must be a 2D array
    - metrics   (list) : the metrics to use for evaluation, must be 'accuracy', 'precision', 'recall', 'f1_score' or 'roc_auc'
    """
    
    for metric in metrics:
      if metric not in Key.METRICS:
        raise ValueError(f"Metric must be one of {list(Key.METRICS.keys())}, got '{metric}'")
    
    return [Key.METRICS[m](labels, [self.predict(feature) for feature in features]) for m in metrics]

class DecisionTree:
  def __init__(self):
    """
    Random Forest
    -----
      Predicts the class of a point based on a decision tree
    """
    self.is_compiled = False
    self.is_trained = False
  
  def compile(self, depth:int, loss:str, split:int, **kwargs):
    """
    Compile
    -----
      Compiles the model to be ready for building.
      'infomration gain' is an alias for 'entropy'. it will be converted to 'entropy' automatically.
    -----
    Args
    -----
    - loss (str)  : the loss function to use
    - split (int) : The minimum number of samples required to split an internal node. If a node has fewer samples than this, it becomes a leaf node.
    - (Optional) max_features (int) : The maximum number of features to consider when looking for the best split, defaults to 'All
    
    Losses
    - Gini
    - Entropy
    
    Max Features
    - All
    - Sqrt
    - Log2
    """
    
    self.depth = depth
    self.loss = loss.lower()
    self.split = split
    self.max_features:str = kwargs.get('max_features', 'all').lower()
    
    self.tree = None
    self.is_compiled = True
    self.is_trained = False
    
    if self.loss not in ['gini', 'entropy']:
      if self.loss == 'information gain':
        self.loss = 'entropy'
      else:
        raise SystemError(f"Loss function must be 'Gini' or 'Entropy' and not {self.loss}")
    
    if self.max_features.lower() not in ('all', 'sqrt', 'log2'):
      raise SystemError(f"Max features must be 'All', 'Sqrt' or 'Log2' and not {self.max_features}")
  
  def fit(self, features:list, labels:list):
    """
    Fit
    -----
      Fits the model to the data, building the decision tree.
    -----
    Args:
    -----
    - features (list) : the features of the dataset, must be an array of coordinates (2D array)
    - labels   (list) : the labels of the dataset, must be an array of classes, it can be of any type
    """
    
    self.is_trained = True
    
    class container(dc):
      """
      data = coordinates
      c = class
      """
      def __init__(self, data, c, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.c = c
      
    class Node(n):
      def __init__(self, data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
    
    # check the data
    if len(shape(features)) != 2:
      raise TypeError(f"Features must be a 2D array, got a {shape(features)}")
    if len(shape(labels)) != 1:
      raise TypeError(f"Labels must be a 1D array, got {shape(features)}")
    if not self.is_compiled:
      raise SystemError("Model must be compiled before fitting")
    
    # setting up variables
    transposed_features = transpose(features)
    purity = Gini_impurity if self.loss == 'gini' else Entropy
    data = [container(f, l) for f,l in zip(features, labels)] # Datacontainer obj, dont forget!
    
    best_feature = 0 # the best feature to split by so far
    best_threshold = 0 # the best threshold so far
    self.tree = Node(data, category=None, threshold=None, decision=None)
    
    leaves = [] # list of leaves to be sliced
    passed_data_left = [] # how much of the data is greater than the threshold
    passed_data_right = [] # how much of the data is lesser than the threshold
    
    stop = False
    
    if self.max_features == 'all':
      max_features = len(transposed_features)
    elif self.max_features == 'sqrt':
      max_features = math.ceil(math.sqrt(len(transposed_features)))
    elif self.max_features == 'log2':
      max_features = math.ceil(math.log2(len(transposed_features)))

    # build the tree uisng BFS
    while not stop:
      
      leaves = self.tree.get_last()
      for leaf in leaves:
        
        impurity = purity([x.c for x in leaf.data])
        
        # iterate over all the features
        
        max_features_valid = [random.randint(0, len(transposed_features) - 1) for _ in range(max_features)]
        
        for feature_index, feature in enumerate(transposed_features):
          
          if feature_index not in max_features_valid and self.max_features != 'all':
            continue
          
          # iterate over all the thresholds possible
          for threshold in feature:
            
            # get the classes of the points that pass and fail the threshold
            passed_data_left = [x.c for x in leaf.data if x.data[feature_index] > threshold]
            passed_data_right = [x.c for x in leaf.data if x.data[feature_index] <= threshold]
            
            # if theres enough data to consider a split
            if len(passed_data_left) > 0:
              
              new_impurity_left = purity(passed_data_left)
              new_impurity_right = purity(passed_data_right)
              new_impurity = (len(passed_data_left) / len(leaf.data)) * new_impurity_left + (len(passed_data_right) / len(leaf.data)) * new_impurity_right
              
              # min value algorithm
              if new_impurity < impurity:
                impurity = new_impurity
                best_feature = feature_index
                best_threshold = threshold
        
        leaf.kwargs['category'] = best_feature      
        leaf.kwargs['threshold'] = best_threshold
        
        # split the node if there is enough data
        if len(leaf.data) >= self.split:
          
          left = Node([x for x in leaf.data if x.data[best_feature] > best_threshold], category=None, threshold=None, decision=None)
          right = Node([x for x in leaf.data if x.data[best_feature] <= best_threshold], category=None, threshold=None, decision=None)
          
          leaf.add_child(left)
          leaf.add_child(right)
        
        if leaf.depth() == self.depth or len(leaf.data) < self.split:
          stop=True
  
    # set classification
    
    for leaf in self.tree.get_last():
      if leaf.kwargs['decision'] is None:
        leaf.kwargs['decision'] = max(set([x.c for x in leaf.data]), key=[x.c for x in leaf.data].count) if len(leaf.data) > 0 else 0
    
  def predict(self, point:list):
    """
    Predict
    -----
      Predicts the class of a point, the model must be compiled before using this method.
    -----
    Args:
    -----
    - point (list) : the point to predict, a vector (list of intigers or floats)
    """
    
    if not self.is_compiled:
      raise SystemError("Model must be compiled before predicting anything")
    if not self.is_trained:
      raise SystemError("Model must be trained before predicting anything")
    
    current_node = self.tree
    
    while True:
      
      # check if the node is a leaf
      if current_node.kwargs['decision'] is not None:
        return current_node.kwargs['decision']
      
      else:
        # check if the category of the point is greater than the threshold
        if point[current_node.kwargs['category']] > current_node.kwargs['threshold']:
          
          # left = bigger & right = smaller
          current_node = current_node.children[0]
        else:
          current_node = current_node.children[1]
  
  def evaluate(self, features, labels, metrics:list):
    """
    Evaluate
    -----
      Evaluates the model on a test set using a specified metric.
    -----
    Args
    -----
    - features (list) : the features of the dataset, must be a 2D array
    - labels   (list) : the labels of the dataset, must be a 2D array
    - metrics   (list) : the metrics to use for evaluation, must be 'accuracy', 'precision', 'recall', 'f1_score' or 'roc_auc'
    """
    
    for metric in metrics:
      if metric not in Key.METRICS:
        raise ValueError(f"Metric must be one of {list(Key.METRICS.keys())}, got '{metric}'")
    
    return [Key.METRICS[m](labels, [self.predict(feature) for feature in features]) for m in metrics]

class RandomForest:
  def __init__(self):
    """
    Random Forest
    -----
      Predicts the class of a point based on a random forest. This model is an ensemble of decision trees.
    """
    self.is_compiled = False
    self.is_trained = False
  
  def compile(self, n_estimators:int, depth:int, loss:str, split:int=0, **kwargs):
    """
    Compile
    -----
      Compiles the model to be ready for building.
      'infomration gain' is an alias for 'entropy'. it will be converted to 'entropy' automatically.
    -----
    Args
    -----
    - n_estimators (int) : the number of trees to build
    - loss (str)  : the loss function to use
    - split (int) : The minimum number of samples required to split an internal node. If a node has fewer samples than this, it becomes a leaf node.
    - depth (int) : The maximum depth of any of the trees
    - (Optional) max_features (str) : The maximum number of features to consider when looking for the best split
    - (Optional) bootstrap (bool) : If True, the bootstrap sample is used when building trees. If False, the whole dataset is used to build each tree. True by default
    
    Losses
    - Gini
    - Entropy 
    
    Max Features
    - All
    - Sqrt
    - Log2
    """
    self.n_estimators = n_estimators
    self.depth = depth
    self.loss = loss.lower()
    self.split = split
    
    self.max_features:str = kwargs.get('max_features', 'all').lower()
    self.bootstrap_data = kwargs.get('bootstrap', True)
    
    self.is_compiled = True
    self.is_trained = False
    self.trees = []
    
    if self.loss not in ['gini', 'entropy']:
      if self.loss == 'information gain':
        self.loss = 'entropy'
      else:
        raise SystemError(f"Loss function must be 'Gini' or 'Entropy' and not '{self.loss}'")
    
    if self.max_features.lower() not in ('all', 'sqrt', 'log2'):
      raise SystemError(f"Max features must be 'All', 'Sqrt' or 'Log2' and not '{self.max_features}'")
  
  def fit(self, features:list, labels:list):
    """
    Fit
    -----
      Fits the model to the data, building the decision tree.
    -----
    Args
    -----
    - features (list) : the features of the dataset, must be an array of coordinates (2D array)
    - labels   (list) : the labels of the dataset, must be an array of classes, it can be of any type
    """
    
    self.is_trained = True
    
    class container(dc):
      """
      data = coordinates
      c = class
      """
      def __init__(self, data, c, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.c = c
    
    # check the data
    if len(shape(features)) != 2:
      raise TypeError(f"Features must be a 2D array, got a {shape(features)}")
    if len(shape(labels)) != 1:
      raise TypeError(f"Labels must be a 1D array, got {shape(features)}")
    if not self.is_compiled:
      raise SystemError("Model must be compiled before fitting")

    # tree maker
    def build_tree(features:list, labels:list, depth:int, loss:str, split:int) -> n:
      """
      Internal class, returns a decision tree
      """
      
      tree = DecisionTree()
      tree.compile(
        depth=depth,
        loss=loss,
        split=split,
        max_features=self.max_features
      )
      tree.fit(features, labels)
      return tree
    
    data = [container(f, l) for f,l in zip(features, labels)] # Datacontainer obj, dont forget!
    data_per_tree = []
    
    # bootstrap data
    if self.bootstrap_data:
      for _ in range(self.n_estimators):
        bootstraped_data = []
        
        for _ in range(len(data)):
          bootstraped_data.append(random.choice(data))
        
        data_per_tree.append(bootstraped_data)
    
    else:
      data_per_tree = [data for _ in range(self.n_estimators)]
    
    for data in data_per_tree:
      tree = build_tree(
        features=[d.data for d in data],
        labels=[d.c for d in data],
        depth=self.depth,
        loss=self.loss,
        split=self.split
      )
      self.trees.append(tree)
    
  def predict(self, point:list):
    """
    Predict
    -----
      Predicts the class of a point, the model must be compiled before using this method.
    -----
    Args:
    -----
    - point (list) : the point to predict, a vector (list of intigers or floats)
    """
    
    if not self.is_compiled:
      raise SystemError("Model must be compiled before predicting anything")
    if not self.is_trained:
      raise SystemError("Model must be trained before predicting anything")
    
    votes = []
    vote_count = {}
    
    for tree in self.trees:
      votes.append(tree.predict(point))
    
    # tally up the votes
    
    for vote in votes:
      if vote in vote_count:
        vote_count[vote] += 1
      else:
        vote_count[vote] = 1
    
    return max(vote_count, key=vote_count.get)

  def evaluate(self, features, labels, metrics:list):
    """
    Evaluate
    -----
      Evaluates the model on a test set using a specified metric.
    -----
    Args
    -----
    - features (list) : the features of the dataset, must be a 2D array
    - labels   (list) : the labels of the dataset, must be a 2D array
    - metrics   (list) : the metrics to use for evaluation, must be 'accuracy', 'precision', 'recall', 'f1_score' or 'roc_auc'
    """
    
    for metric in metrics:
      if metric not in Key.METRICS:
        raise ValueError(f"Metric must be one of {list(Key.METRICS.keys())}, got '{metric}'")
    
    return [Key.METRICS[m](labels, [self.predict(feature) for feature in features]) for m in metrics]

class NaiveBayes:
  def __init__(self):
    """
    Naive Bayes
    -----
      Predicts the class of a point based using bayes's theorem assuming the data is independent of each other.
    """
    self.is_compiled = False
    self.is_trained = False

  def compile(self, model_type:str, return_type:str="argmax", **kwargs):
    """
    Compile
    -----
      Compiles the model, model type must be 'Gaussian' for continous data, 'Multinomial' for discrete data or 'Bernoulli' for binary data
    -----
    Args
    -----
    - model_type (str) : the type of model to compile, must be 'Gaussian', 'Multinomial' or 'Bernoulli'
    - return_type (str) : the type of return value, must be 'log', 'probability' or 'argmax', defaults to 'argmax'
    - (Optional) laplace_smoothing (int) : the laplace smoothing to use for the multinomial and bernoulli models, defaults to 1
    """
    
    self.is_compiled = True
    self.type = model_type
    self.return_type = return_type.lower()
    self.laplace_smoothing = kwargs.get('laplace_smoothing', 1)
    
    if model_type not in ("gaussian", "multinomial", "bernoulli"):
      raise SystemError(f"Model type must be 'Gaussian', 'Multinomial' or 'Bernoulli' and not '{model_type}'")
    if self.return_type not in ("log", "probability", "argmax"):
      raise SystemError(f"Return type must be 'log', 'probability', 'argmax' or 'softmax' and not '{self.return_type}'")
  
  def fit(self, features, labels):
    """
    Fit
    -----
      Calculates the necessary statistics to make predictions to be used later in the predict method. Make sure the data
      aligns with the model type.
    -----
    Args
    -----
    - features (list) : the features of the dataset, must be an array of coordinates (2D array)
    - labels   (list) : the labels of the dataset, must be an array of classes, it can be of any type
    """
    self.is_trained = True
    self.unique_classes = set(labels)
    self.cached_data = []
    
    if self.type == "gaussian":
      # p(x|y) = (1 / sqrt(2 * pi * var)) * exp(- (x - mean)^2 / (2 * var))
      # var = sum((x - mean)^2) / len(x)
      # mean = sum(x) / len(x)
      # variance is just the standard deviation squared
      
      # find the mean and variance of each feature per class
      
      # cached_data stores a per-class data containing [prior, [means, variances]],
      
      for unique_class in self.unique_classes:
        
        prior_probability = math.log(labels.count(unique_class) / len(labels))
        
        current_class_features = []
        for feature, label in zip(features, labels):
          if label == unique_class:
            current_class_features.append(feature)
        
        current_class_feature_stats = []
        for cols in transpose(current_class_features):
          mean = sum(cols) / len(cols)
          variance = sum((x - mean) ** 2 for x in cols) / len(cols)
          current_class_feature_stats.append([mean, variance])
        
        self.cached_data.append([prior_probability, current_class_feature_stats])
      
      pass
    
    elif self.type == "bernoulli":
      
      for feature in features:
        for value in feature:
          if value not in (0, 1):
            raise ValueError(f"Bernoulli model only supports binary data (0 and 1), got {value}")
      
      # cashed_data stores a per-class data containing [count of Y,count of Y, feature counts]
      
      for unique_class in self.unique_classes:
        
        current_class_features = []
        for feature, label in zip(features, labels):
          if label == unique_class:
            current_class_features.append(feature)
        
        current_class_feature_count = []
        for cols in transpose(current_class_features):
          current_class_feature_count.append(sum(cols))
        
        self.cached_data.append([len(current_class_features), current_class_feature_count])
    
    elif self.type == "multinomial":
      # cached_data stores a per-class data containing [prior, feature counts]
      
      for unique_class in self.unique_classes:
        
        prior_probability = math.log(labels.count(unique_class) / len(labels))
        
        current_class_features = []
        for feature, label in zip(features, labels):
          if label == unique_class:
            current_class_features.append(feature)
        
        current_class_feature_count = []
        for cols in transpose(current_class_features):
          current_class_feature_count.append(sum(cols))
        
        self.cached_data.append([prior_probability, current_class_feature_count])
      
  def predict(self, point):
    """
    Predict
    -----
      Predicts the class of a point, the model must be compiled and trained before using this method.
    -----
    Args:
    -----
    - point (list) : the point to predict, a vector (list of intigers or floats)
    """
    
    log_class_probabilities = []
    
    if self.type == "gaussian":
      
      for prior_probability, current_class_feature_stats in self.cached_data:
        
        log_likelyhood = 0
        for i in range(len(point)):
          mean, variance = current_class_feature_stats[i]
          
          exponent = math.exp(- ((point[i] - mean) ** 2) / (2 * variance)) if variance > 0 else 0
          likelihood = (1 / math.sqrt(2 * math.pi * variance)) * exponent if variance > 0 else 0
          
          log_likelyhood += math.log(likelihood) if likelihood > 0 else -1000 # prevent log(0)
        
        log_class_probabilities.append(prior_probability + log_likelyhood)
        
    elif self.type == "bernoulli":
      
      for class_count, current_class_feature_count in self.cached_data:
        
        prior_probability = math.log(class_count / sum(c[0] for c in self.cached_data))
        denominator = class_count + self.laplace_smoothing * 2
        
        log_likelyhood = 0
        for i in range(len(point)):
          numerator = current_class_feature_count[i] + self.laplace_smoothing
          probability = numerator / denominator
          
          log_likelyhood += math.log(probability) if point[i] == 1 else math.log(1 - probability)
        
        log_class_probabilities.append(prior_probability + log_likelyhood)
    
    elif self.type == "multinomial":
      
      for prior_probability, current_class_feature_count in self.cached_data:
        
        log_likelyhood = 0
        for i in range(len(point)):
          numerator = current_class_feature_count[i] + self.laplace_smoothing
          denominator = sum(current_class_feature_count) + self.laplace_smoothing * len(current_class_feature_count)
          log_likelyhood += math.log(numerator / denominator) * point[i]
        
        log_class_probabilities.append(prior_probability + log_likelyhood)
    
    if self.return_type == "log":
      return log_class_probabilities
    elif self.return_type == "probability":
      max_log = max(log_class_probabilities)
      exp_probs = [math.exp(lp - max_log) for lp in log_class_probabilities] # prevent overflow
      sum_exp_probs = sum(exp_probs)
      return [p / sum_exp_probs for p in exp_probs]
    elif self.return_type == "argmax":
      return list(self.unique_classes)[log_class_probabilities.index(max(log_class_probabilities))]

  def evaluate(self, features, labels, metrics:list):
    """
    Evaluate
    -----
      Evaluates the model on a test set using a specified metric.
    -----
    Args
    -----
    - features (list) : the features of the dataset, must be a 2D array
    - labels   (list) : the labels of the dataset, must be a 2D array
    - metrics   (list) : the metrics to use for evaluation, must be 'accuracy', 'precision', 'recall', 'f1_score' or 'roc_auc'
    """
    
    for metric in metrics:
      if metric not in Key.METRICS:
        raise ValueError(f"Metric must be one of {list(Key.METRICS.keys())}, got '{metric}'")
    
    return [Key.METRICS[m](labels, [self.predict(feature) for feature in features]) for m in metrics]

class SVM:
  def __init__(self):
    """
    Support Vector Machine
    -----
      Predicts the class of a point based on a boundary. Note that SVMs only support binary classification, for multiclass
      classification, use the MSVM class.
    """
    self.alphas = []
    self.b = 0
    self.is_compiled = False
    self.is_trained = False
  
  def compile(self, kernel:str, maximum_iterations:int, learning_rate:float|int, *args, **kwargs):
    """
    Compile
    -----
      Compiles the model to be ready for training.
    -----
    Args
    -----
    - kernel                    (core.vanilla.kernels object)  : must be a core.vanilla.kernel object, if not, make sure it contains a __call__ method
    - maximum_iterations        (int)                          : the maximum number of iterations to train the model for to prevent infinite loops
    - learning_rate             (float)                        : the learning rate to use when training the model
    - (Optional) c              (float)                        : the regularization parameter
    - (Optional) l2lambda       (float)                        : the L2 regularization parameter
    - (Optional) return_scores  (bool)                         : if True, the predict method will return the score instead of the class, defaults to False
    """
    self.kernel = kernel
    self.learning_rate = learning_rate
    self.b = 0
    self.max_iter = maximum_iterations
    self.classes = set()
    self.is_compiled = True
    self.c = kwargs.get('c', 1E+10) # regularization parameter
    self.l2lambda = kwargs.get('l2lambda', 0.01) # L2 regularization parameter
    self.return_scores = kwargs.get('return_scores', False) # if True, the predict method will return the score instead of the class
    
  def fit(self, features:list, labels:list):
    """
    Fit
    -----
      Fits the model to the data, training the support vector machine.
    -----
    Args:
    -----
    - features (list) : the features of the dataset, must be a 2D array
    - labels   (list) : the labels of the dataset, must be a 1D array
    """
    
    if not self.is_compiled:
      raise SystemError("Model must be compiled before fitting")
    
    # check and normalize data
    if len(shape(features)) != 2:
      raise TypeError(f"Features must be a 2D array, got {shape(features)}")
    if len(shape(labels)) != 1:
      raise TypeError(f"Labels must be a 1D array, got {shape(labels)}")
    
    self.classes = set(labels)
    if len(self.classes) != 2:
      raise ValueError(f"SVM only supports binary classification, got {len(self.classes)} classes")
    self.labels = [1 if x == list(self.classes)[0] else -1 for x in labels] # convert to 1 and -1
    self.features = features
    
    self.alphas = [0] * len(features)
    self.b = 0
    
    for _ in range(self.max_iter):
      errors = 0
      
      # L2 decay step
      self.alphas = [max(alpha * ( 1 - self.learning_rate * self.l2lambda), 0) for alpha in self.alphas]
      
      for i in range(len(features)):
        x_i = features[i]
        y_i = self.labels[i]
        score = 0
        
        # Iterate over all points
        for alpha, x_j, y_j in zip(self.alphas, self.features, self.labels):
          if alpha > 0:
            kernel_output = self.kernel(x_i, x_j)
            score += alpha * y_j * kernel_output
                
        score += self.b # final score with bias
        
        if y_i * score <= 1: # 2. check for misclassification
          errors += 1
          
          self.alphas[i] += self.learning_rate # Dual Update Rule
          self.b += y_i * self.learning_rate # Update bias
          
          self.alphas = [min(alpha, self.c) for alpha in self.alphas] # clip alphas to C
            
      if errors == 0:
        break
      
    self.is_trained = True
    
  def predict(self, point:list):
    """
    Predict
    -----
      Predicts the class of a point, the model must be compiled before using this method.
    -----
    Args
    -----
    - point (list) : the point to predict, a vector (list of intigers or floats)
    """
    if not self.is_compiled:
      raise SystemError("Model must be compiled before predicting anything")
    if not self.is_trained:
      raise SystemError("Model must be trained before predicting anything")
    
    score = 0
    # Iterate over all points
    for alpha, x_j, y_j in zip(self.alphas, self.features, self.labels):

      kernel_output = self.kernel(x_j, point)
      score += alpha * y_j * kernel_output
    
    if self.return_scores:
      return score + self.b
    
    return list(self.classes)[0] if score + self.b > 0 else list(self.classes)[1]

  def evaluate(self, features, labels, metrics:list):
    """
    Evaluate
    -----
      Evaluates the model on a test set using a specified metric.
    -----
    Args
    -----
    - features (list) : the features of the dataset, must be a 2D array
    - labels   (list) : the labels of the dataset, must be a 2D array
    - metrics   (list) : the metrics to use for evaluation, must be 'accuracy', 'precision', 'recall', 'f1_score' or 'roc_auc'
    """
    
    for metric in metrics:
      if metric not in Key.METRICS:
        raise ValueError(f"Metric must be one of {list(Key.METRICS.keys())}, got '{metric}'")
    
    return [Key.METRICS[m](labels, [self.predict(feature) for feature in features]) for m in metrics]

class MSVM:
  def __init__(self):
    """
    Multiclass Support Vector Machine
    -----
      Predicts the class of a point based on a boundary. Works by using multiple SVMs to classify the data using the
      OvA algorithm.
    """
    self.alphas = []
    self.b = 0
    self.is_compiled = False
    self.is_trained = False
    self.models = []
  
  def compile(self, kernel:str, maximum_iterations:int, learning_rate:float|int, *args, **kwargs):
    """
    Compile
    -----
      Compiles the model to be ready for training.
    -----
    Args
    -----
    - kernel                    (core.vanilla.kernels object)  : must be a core.vanilla.kernel object, if not, make sure it contains a __call__ method
    - maximum_iterations        (int)                          : the maximum number of iterations to train the model for to prevent infinite loops
    - learning_rate             (float)                        : the learning rate to use when training the model
    - (Optional) c              (float)                        : the regularization parameter
    - (Optional) l2lambda       (float)                        : the L2 regularization parameter
    - (Optional) return_scores  (bool)                         : if True, the predict method will return the score instead of the class, defaults to False
    """
    # might just implement this later, for now stick to OvA
    # - type                (str)     : the type of multiclass SVM to use, must be 'OvO' (one-vs-one) or 'OvA' (one-vs-all)
    
    self.kernel = kernel
    self.learning_rate = learning_rate
    self.b = 0
    self.max_iter = maximum_iterations
    self.classes = set()
    self.is_compiled = True
    self.c = kwargs.get('c', 1E+10) # regularization parameter
    self.l2lambda = kwargs.get('l2lambda', 0.01) # L2 regularization parameter
    self.return_scores = kwargs.get('return_scores', False) # if True, the predict method will return the score instead of the class
  
  def fit(self, features, labels):
    """
    Fit
    -----
      Fits the model to the data, training the support vector machine.
    -----
    Args:
    -----
    - features (list) : the features of the dataset, must be a 2D array
    - labels   (list) : the labels of the dataset, must be a 1D array
    """
    if not self.is_compiled:
      raise SystemError("Model must be compiled before fitting")
    
    # check and normalize data
    if len(shape(features)) != 2:
      raise TypeError(f"Features must be a 2D array, got {shape(features)}")
    if len(shape(labels)) != 1:
      raise TypeError(f"Labels must be a 1D array, got {shape(labels)}")
    
    # convert data to ova format
    self.models = []
    self.classes = set(labels)
    
    for current_class in self.classes:
      binary_labels = [1 if x == current_class else -1 for x in labels]
      
      model = SVM()
      model.compile(self.kernel, self.max_iter, self.learning_rate, c=self.c, l2lambda=self.l2lambda, return_scores=True)
      model.fit(features, binary_labels)
      
      self.models.append(model)
    
    self.is_trained = True
  
  def predict(self, point):
    """
    Predict
    -----
      Predicts the class of a point, the model must be compiled before using this method.
    -----
    Args
    -----
    - point (list) : the point to predict, a vector (list of intigers or floats)
    """
    if not self.is_compiled:
      raise SystemError("Model must be compiled before predicting anything")
    if not self.is_trained:
      raise SystemError("Model must be trained before predicting anything")
    
    scores = [model.predict(point) for model in self.models]
    if self.return_scores:
      return scores
    return list(self.classes)[scores.index(max(scores))]

  def evaluate(self, features, labels, metrics:list):
    """
    Evaluate
    -----
      Evaluates the model on a test set using a specified metric.
    -----
    Args
    -----
    - features (list) : the features of the dataset, must be a 2D array
    - labels   (list) : the labels of the dataset, must be a 2D array
    - metrics   (list) : the metrics to use for evaluation, must be 'accuracy', 'precision', 'recall', 'f1_score' or 'roc_auc'
    """
    
    for metric in metrics:
      if metric not in Key.METRICS:
        raise ValueError(f"Metric must be one of {list(Key.METRICS.keys())}, got '{metric}'")
    
    return [Key.METRICS[m](labels, [self.predict(feature) for feature in features]) for m in metrics]
