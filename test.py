import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import api.staticnet as net

from tools.arraytools import generate_random_array
import matplotlib.pyplot as plt
import time
import math
import random
import numpy as np
from core.static.encoders import OneHotEncode
from tools.visual import display_boundary

start_time = time.perf_counter()
test = 7

if test == 1: # CNN

  up = [
    [1,1,1],
    [0,0,0],
    [0,0,0]
  ]
  down = [
    [0,0,0],
    [0,0,0],
    [1,1,1]
  ]
  left = [
    [1,0,0],
    [1,0,0],
    [1,0,0]
  ]
  right = [
    [0,0,1],
    [0,0,1],
    [0,0,1]
  ]
  
  trainX = [up, down, left, right]
  trainY = [
    [1,1,0,0], 
    [0,0,1,1], 
    [1,0,1,0], 
    [0,1,0,1]
    ]
  
  model = net.Sequential( 
    
    net.Convolution((2,2), 3, 'relu'),
    net.Flatten(),
  )

  model.compile(
    optimizer='default', 
    loss='mean squared error', 
    learning_rate=0.1, 
    epochs=10000,
    metrics=['accuracy']
  )
  
  model.fit(
    trainX, 
    trainY, 
  )

  print(model.push(up))
  # print([1,0])

elif test == 2: # NN
  training_features = [[0,0],[0,1],[1,0],[1,1]]
  training_target   = [[0,1],[1,0],[1,0],[0,1]]

  model = net.Sequential(
    net.Dense(15, activation='elu'),
    net.Dense(2, activation='elu'),
  )

  model.compile(
    optimizer='none', 
    loss='mean squared error', 
    learning_rate=0.01,
    batch_size=1,
    epochs=1000, 
    metrics=['accuracy'],
    experimental=['track metrics'],
    verbose=4,
    logging = 100
  )
  
  model.fit(
    training_features, 
    training_target, 
  )
  
  model.evaluate(
    training_features, 
    training_target,
    verbose=1,
    logging=True
  )
  
  for feature, target in zip(training_features, training_target):
    print(f"pred {model.push(feature)} true {target}")
  
  display_boundary(model, training_features, training_target)
  
elif test == 3: # RNN
  training_features = [
    
    [1, 2, 3, 4],   # 10
    [0, 1, 2, 3],   # 6
    [5, 5, 0, 0],   # 10
    [1, 1, 1, 1],   # 4
    [2, 2, 1, 2]    # 7
  ]

  training_target = [
    [10],
    [6],
    [10],
    [4],
    [7]
  ]

  model = net.Sequential(
    net.Recurrent('none', input=True, output=False),
    net.Recurrent('none', input=True, output=False),
    net.Recurrent('none', input=True, output=False),
    net.Recurrent('none', input=True, output=False),
    net.Recurrent('none', input=False, output=True),
  )
  
  model.compile(
    optimizer='rmsprop', 
    loss='mean squared error', 
    learning_rate=0.001,
    batch_size=3, 
    epochs=10000,
    metrics=['accuracy']
  ) 
  
  model.fit(
    training_features, 
    training_target, 
    )
  
  for feature, target in zip(training_features, training_target):
    print(f"pred {model.push(feature)} true {target}")
    
  print(f"{model.push([5,5,5,5])} true [20]")

elif test == 4: # LSTM
  training_features = [
    
    [1, 2, 3, 4],   # 10
    [0, 1, 2, 3],   # 6
    [5, 5, 0, 0],   # 10
    [1, 1, 1, 1],   # 4
    [2, 2, 1, 2]    # 7
  ]

  training_target = [
    [1],
    [-1/3],
    [1],
    [-1],
    [0]
  ]

  model = net.Sequential(
    net.LSTM(input=True, output = False),
    net.LSTM(input=True, output = False),
    net.LSTM(input=True, output = False),
    net.LSTM(input=True, output = False),
    net.LSTM(input=False, output = True),
  )
  
  model.compile(
    optimizer='rmsprop', 
    loss='mean squared error', 
    learning_rate=0.0001,
    batch_size=3, 
    epochs=50000,
    metrics=['accuracy']
  ) 
  
  model.fit(
    training_features, 
    training_target, 
    )
   
  for feature, target in zip(training_features, training_target):
    print(f"pred {model.push(feature)} true {target}")

elif test == 5: # GRU
  training_features = [
    
    [1, 2, 3, 4],   # 10
    [0, 1, 2, 3],   # 6
    [5, 5, 0, 0],   # 10
    [1, 1, 1, 1],   # 4
    [2, 2, 1, 2]    # 7
  ]

  training_target = [
    [1],
    [-1/3],
    [1],
    [-1],
    [0]
  ]

  model = net.Sequential(
    net.GRU(input=True, output = False),
    net.GRU(input=True, output = False),
    net.GRU(input=True, output = False),
    net.GRU(input=True, output = False),
    net.GRU(input=False, output = True),
  )
  
  model.compile(
    optimizer='adam', 
    loss='mean squared error', 
    learning_rate=0.001,
    batch_size=1, 
    epochs=50000,
    metrics=['accuracy']
  )  
  
  model.fit(
    training_features, 
    training_target, 
    )
   
  for feature, target in zip(training_features, training_target):
    print(f"pred {model.push(feature)} true {target}")

elif test == 7: # 1D test

  def dataset(num_samples, input_dimensions, output_dimensions, noise_strength):
    """
    Generates synthetic noisy data for linear regression using pure Python lists.

    Args:
        num_samples (int): The number of data points to generate.
        input_dimensions (int): The number of features for each input sample.
        output_dimensions (int): The number of target values for each output sample.
                                  (Default: 1 for simple linear regression)
        noise_strength (float): The magnitude of random noise added to the outputs.
                                  A higher value means more noise.

    Returns:
        tuple[list, list]: A tuple containing two lists:
                            - inputs (list of lists): [[x1, x2, ...], [x1, x2, ...], ...]
                            - outputs (list of lists): [[y1, y2, ...], [y1, y2, ...], ...]
    """
    inputs = []
    outputs = []

    # Generate random "true" weights and biases for the underlying linear relationship
    # These are the parameters your linear regression model would try to learn
    true_weights = [
      [random.uniform(-1, 1) for _ in range(output_dimensions)]
      for _ in range(input_dimensions)
    ]
    true_biases = [random.uniform(-0.5, 0.5) for _ in range(output_dimensions)]

    for _ in range(num_samples):
      # Generate a random input sample
      current_input = [random.uniform(-20, 20) for _ in range(input_dimensions)]
      inputs.append(current_input)

      # Calculate the true linear output before adding noise
      current_true_output = [0.0] * output_dimensions
      for out_dim in range(output_dimensions):
        dot_product = 0.0
        for in_dim in range(input_dimensions):
          dot_product += current_input[in_dim] * true_weights[in_dim][out_dim]
        current_true_output[out_dim] = dot_product + true_biases[out_dim]

      # Add noise to the output
      current_noisy_output = [
        val + random.uniform(-noise_strength, noise_strength)
        for val in current_true_output
      ]
      outputs.append(current_noisy_output)

    return inputs, outputs

  def binary_dataset(num_samples: int, input_dimensions: int, output_dimensions: int, noise_strength: float):
      """
      Generates synthetic noisy binary classification data suitable for logistic regression.

      The 'true' underlying relationship is linear in the inputs, transformed by a sigmoid,
      and then thresholded to binary (0 or 1) with noise influencing the decision boundary.

      Args:
          num_samples (int): The number of data points to generate.
          input_dimensions (int): The number of features for each input sample.
          output_dimensions (int): The number of binary target values for each output sample.
                                  (Typically 1 for standard binary classification,
                                    but can be >1 for multi-label binary classification where
                                    each output is an independent binary classification).
          noise_strength (float): The magnitude of random noise added to the *logit*
                                  (the linear score before the sigmoid). A higher value
                                  makes the classes less separable (more overlap).

      Returns:
          tuple[list, list]: A tuple containing two lists:
                            - inputs (list of lists): [[x1, x2, ...], [x1, x2, ...], ...]
                            - outputs (list of lists): [[0 or 1], [0 or 1], ...]
                              (Each inner list will have `output_dimensions` elements).
      """
      inputs = []
      outputs = []

      # Generate random "true" weights and biases for the underlying linear relationship
      # These parameters define the true decision boundary for each output dimension
      true_weights = [
          [random.uniform(-1, 1) for _ in range(output_dimensions)]
          for _ in range(input_dimensions)
      ]
      true_biases = [random.uniform(-0.5, 0.5) for _ in range(output_dimensions)]

      for _ in range(num_samples):
          # Generate a random input sample (features)
          current_input = [random.uniform(-5, 5) for _ in range(input_dimensions)]
          inputs.append(current_input)

          current_binary_output = [0] * output_dimensions # Initialize list for current sample's binary outputs
          
          for out_dim in range(output_dimensions):
              # Calculate the linear score (logit) for the current output dimension
              logit = 0.0
              for in_dim in range(input_dimensions):
                  logit += current_input[in_dim] * true_weights[in_dim][out_dim]
              logit += true_biases[out_dim]

              # Add noise to the logit. This makes the probability 'fuzzy' around the decision boundary.
              noisy_logit = logit + random.uniform(-noise_strength, noise_strength)

              # Apply the sigmoid function to convert the noisy logit into a probability (0 to 1)
              probability = 1.0 / (1.0 + math.exp(-noisy_logit))

              # Threshold the probability to get a binary class (0 or 1)
              # If the probability is 0.5 or higher, classify as 1; otherwise, 0.
              predicted_class = 1 if probability >= 0.5 else 0
              
              current_binary_output[out_dim] = predicted_class
          
          outputs.append(current_binary_output) # Add the list of binary outputs for this sample

      return inputs, outputs

  training_features, training_target = dataset(100, 1, 1, 5.3)
  
  # training_features = [
  #                       [ (x/1)-15 ] 
  #                       for x in range(30) ]
  # training_target   = [ 
  #                       [ x[0]**3 + 4*x[0]**2 - 2*x[0] ] 
  #                       for x in training_features ]
  
  model = net.Sequential(
    net.Dense(10, 'leaky relu'),
    net.Dense(1, 'none'),
  )

  model.compile(
    optimizer='adam', 
    loss='mean squared error', 
    learning_rate=0.001,
    batch_size=1,
    epochs=1000, 
    metrics=['accuracy'],
    logging=10,
    verbose=4
  )

  model.fit(
    training_features,
    training_target
  )

  testing_domain = [(x/10)-20 for x in range(400)]
  predicted_values = [model.push([x]) for x in testing_domain]

elif test == 8: # 2D test
  
  from sklearn.datasets import make_moons, make_blobs

  def plot_decision_boundary(model, X_data, y_data, step_size):
    """
    Plots the decision boundary of a 2D classification model.
    This function specifically handles models that:
    1. Accept input data as a list of lists (e.g., [[f1_1, f2_1], [f1_2, f2_2], ...]).
    2. Output predictions as a list (e.g., [0, 1, 0, ...] or [0.1, 0.9, 0.2, ...]).

    Args:
        model: A trained classification model instance with a 'predict' method.
               The 'predict' method should conform to the list-in/list-out interface.
        X_data (list): A list of data points, where each inner list is [feature1, feature2].
                       (e.g., [[1.2, 0.5], [-0.3, 2.1], ...])
        y_data (list): A list of true labels corresponding to X_data.
                       (e.g., [0, 1, 0, ...])
        title (str): Title for the plot.
    """
    # 1. Convert input lists to NumPy arrays for easier manipulation with numpy/matplotlib
    X_np = np.array(X_data)
    y_np = np.array(y_data)

    if X_np.shape[1] != 2:
      print("Warning: This function is best for 2-feature data. Plotting first two features.")
      X_plot = X_np[:, :2] # Use only the first two features for plotting
    else:
      X_plot = X_np

    x_min, x_max = X_plot[:, 0].min() - 0.5, X_plot[:, 0].max() + 0.5
    y_min, y_max = X_plot[:, 1].min() - 0.5, X_plot[:, 1].max() + 0.5

    # Create a dense grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), # Step size determines resolution
                         np.arange(y_min, y_max, step_size))
 
    grid_points_flat_np = np.c_[xx.ravel(), yy.ravel()]
    grid_points_for_model = grid_points_flat_np.tolist()
    
    raw_predictions_list = [model.push(point) for point in grid_points_for_model]
    raw_predictions_np = np.array(raw_predictions_list)

    # 6. Interpret model output (probabilities vs. hard labels) and convert to integer class labels
    Z = None
    unique_true_classes = 2
    
    if raw_predictions_np.ndim == 1:
      # Case A: Binary classification with single probability output (e.g., [0.1, 0.9, 0.4])
      if np.issubdtype(raw_predictions_np.dtype, np.floating) and np.all((raw_predictions_np >= 0) & (raw_predictions_np <= 1)):
        Z = (raw_predictions_np > 0.5).astype(int) # Threshold probabilities at 0.5
      # Case B: Already hard labels for binary (e.g., [0, 1, 0])
      else:
        Z = raw_predictions_np.astype(int)
    elif raw_predictions_np.ndim == 2:
      # Case C: Multi-class probabilities (e.g., [[0.1,0.8,0.1],[0.7,0.2,0.1]])
      if np.issubdtype(raw_predictions_np.dtype, np.floating) and np.all((raw_predictions_np >= 0) & (raw_predictions_np <= 1)):
        Z = np.argmax(raw_predictions_np, axis=1) # Get the index of the highest probability (the class label)
      # Case D: Already one-hot encoded or multi-output hard labels (less common for this plotting type)
      # For simplicity, we'll assume argmax is the right way if 2D float output
      else:
        Z = np.argmax(raw_predictions_np, axis=1) # This handles common cases like one-hot hard labels too
    else:
      raise ValueError("Model output format not recognized. Expected 1D or 2D array for probabilities or hard labels.")
    
    # Reshape the predictions back to the grid shape
    Z = Z.reshape(xx.shape)

    # 7. Plot the contour/decision regions
    plt.figure(figsize=(9, 7))

    # Choose colormap based on number of unique classes
    num_classes = 2
    if num_classes <= 2:
      cmap_regions = plt.cm.RdYlBu # Good for binary
      cmap_points = plt.cm.RdYlBu
    else:
      # For multi-class, use a colormap that provides distinct colors for each class
      cmap_regions = plt.cm.get_cmap('viridis', num_classes)
      cmap_points = plt.cm.get_cmap('viridis', num_classes) # Or 'tab10', 'Set1' etc.
    
    # Define levels to ensure clear boundaries for integer classes
    levels = np.arange(np.min(Z), np.max(Z) + 2) - 0.5 # E.g., for classes 0,1,2, levels will be -0.5, 0.5, 1.5, 2.5

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_regions, levels=levels)

    # 8. Overlay the original data points
    COLORS = ['r' if C[0] == 0 else 'b' for C in y_data]
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=COLORS, cmap=cmap_points,
                edgecolors='k', s=30, label="True Labels")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
  
  training_features, training_target = make_blobs(n_samples=200, n_features=2, centers=3, random_state=0, cluster_std=0.9)

  training_features = training_features.tolist()
  training_target = [ OneHotEncode(3, x)  for x in training_target.tolist()]
  
  model = net.Sequential(
    net.Dense(2, 'leaky relu'),
    net.Dense(3, 'leaky relu'),
    net.Dense(3, 'none'),
    net.Operation('softmax')
  )
  
  model.compile(
    optimizer='adam',
    loss='binary crossentropy',
    learning_rate=0.001,
    batch_size=1,
    epochs=500,
    metrics=['accuracy'],
    logging=10,
    verbose=1
  )
  
  model.fit(
    training_features,
    training_target
  )

end_time = time.perf_counter()
duration = end_time - start_time
print(f"""
      finished training in {duration} seconds
      """)

# visualization and diagnosing

if test == 7:
  plt.subplot(2, 1, 1)
  plt.plot(range(len(model.error_logs))           , model.error_logs           , color='red')
  plt.plot(range(len(model.validation_error_logs)), model.validation_error_logs, color='blue')

  plt.subplot(2, 1, 2)
  plt.scatter(training_features, training_target, color='black', s=20)
  plt.plot(testing_domain, predicted_values, color='red')
  plt.show()
  
if test == 8:
  plt.plot(range(len(model.error_logs))           , model.error_logs           , color='red')
  plt.plot(range(len(model.validation_error_logs)), model.validation_error_logs, color='blue')
  
  display_boundary(model, training_features, training_target)