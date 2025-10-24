import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
from models.regressor import *
import random
import time

# def dataset(num_samples, input_dimensions, output_dimensions, noise_strength):
#   """
#   Generates synthetic noisy data for linear regression using pure Python lists.

#   Args:
#       num_samples (int): The number of data points to generate.
#       input_dimensions (int): The number of features for each input sample.
#       output_dimensions (int): The number of target values for each output sample.
#                                 (Default: 1 for simple linear regression)
#       noise_strength (float): The magnitude of random noise added to the outputs.
#                                 A higher value means more noise.

#   Returns:
#       tuple[list, list]: A tuple containing two lists:
#                           - inputs (list of lists): [[x1, x2, ...], [x1, x2, ...], ...]
#                           - outputs (list of lists): [[y1, y2, ...], [y1, y2, ...], ...]
#   """
#   inputs = []
#   outputs = []

#   # Generate random "true" weights and biases for the underlying linear relationship
#   # These are the parameters your linear regression model would try to learn
#   true_weights = [
#     [random.uniform(-1, 1) for _ in range(output_dimensions)]
#     for _ in range(input_dimensions)
#   ]
#   true_biases = [random.uniform(-0.5, 0.5) for _ in range(output_dimensions)]

#   for _ in range(num_samples):
#     # Generate a random input sample
#     current_input = [random.uniform(-5, 5) for _ in range(input_dimensions)]
#     inputs.append(current_input)

#     # Calculate the true linear output before adding noise
#     current_true_output = [0.0] * output_dimensions
#     for out_dim in range(output_dimensions):
#       dot_product = 0.0
#       for in_dim in range(input_dimensions):
#           dot_product += current_input[in_dim] * true_weights[in_dim][out_dim]
#       current_true_output[out_dim] = dot_product + true_biases[out_dim]

#     # Add noise to the output
#     current_noisy_output = [
#       val + random.uniform(-noise_strength, noise_strength)
#       for val in current_true_output
#     ]
#     outputs.append(current_noisy_output)

#   return inputs, outputs

# def binary_dataset(num_samples: int, input_dimensions: int, output_dimensions: int, noise_strength: float):
#     """
#     Generates synthetic noisy binary classification data suitable for logistic regression.

#     The 'true' underlying relationship is linear in the inputs, transformed by a sigmoid,
#     and then thresholded to binary (0 or 1) with noise influencing the decision boundary.

#     Args:
#         num_samples (int): The number of data points to generate.
#         input_dimensions (int): The number of features for each input sample.
#         output_dimensions (int): The number of binary target values for each output sample.
#                                  (Typically 1 for standard binary classification,
#                                   but can be >1 for multi-label binary classification where
#                                   each output is an independent binary classification).
#         noise_strength (float): The magnitude of random noise added to the *logit*
#                                 (the linear score before the sigmoid). A higher value
#                                 makes the classes less separable (more overlap).

#     Returns:
#         tuple[list, list]: A tuple containing two lists:
#                            - inputs (list of lists): [[x1, x2, ...], [x1, x2, ...], ...]
#                            - outputs (list of lists): [[0 or 1], [0 or 1], ...]
#                              (Each inner list will have `output_dimensions` elements).
#     """
#     inputs = []
#     outputs = []

#     # Generate random "true" weights and biases for the underlying linear relationship
#     # These parameters define the true decision boundary for each output dimension
#     true_weights = [
#         [random.uniform(-1, 1) for _ in range(output_dimensions)]
#         for _ in range(input_dimensions)
#     ]
#     true_biases = [random.uniform(-0.5, 0.5) for _ in range(output_dimensions)]

#     for _ in range(num_samples):
#         # Generate a random input sample (features)
#         current_input = [random.uniform(-5, 5) for _ in range(input_dimensions)]
#         inputs.append(current_input)

#         current_binary_output = [0] * output_dimensions # Initialize list for current sample's binary outputs
        
#         for out_dim in range(output_dimensions):
#             # Calculate the linear score (logit) for the current output dimension
#             logit = 0.0
#             for in_dim in range(input_dimensions):
#                 logit += current_input[in_dim] * true_weights[in_dim][out_dim]
#             logit += true_biases[out_dim]

#             # Add noise to the logit. This makes the probability 'fuzzy' around the decision boundary.
#             noisy_logit = logit + random.uniform(-noise_strength, noise_strength)

#             # Apply the sigmoid function to convert the noisy logit into a probability (0 to 1)
#             probability = 1.0 / (1.0 + math.exp(-noisy_logit))

#             # Threshold the probability to get a binary class (0 or 1)
#             # If the probability is 0.5 or higher, classify as 1; otherwise, 0.
#             predicted_class = 1 if probability >= 0.5 else 0
            
#             current_binary_output[out_dim] = predicted_class
        
#         outputs.append(current_binary_output) # Add the list of binary outputs for this sample

#     return inputs, outputs

start_time_py = time.perf_counter()

# a * b^c+x + d
a = 1
b = 1 
d = 0

X, Y = [[x] for x in range(10)], [[a * 2.71828**(b*x) + d] for x in range(10)]

model = Exponential(1, 1)
model.compile(
  optimizer='adam',
  loss='mean squared error',
  learning_rate=0.0001,
  epochs=50000,
)

model.fit(
  X,
  Y,
  verbose=4,
  regularity=1000
)

end_time_py = time.perf_counter()
print(f"Ran in {end_time_py - start_time_py:.4f} seconds")

predicted_line = [model.predict([x]) for x in range(10)]

plt.plot(range(10), predicted_line)
plt.scatter(X, Y)
plt.show()