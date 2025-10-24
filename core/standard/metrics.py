import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from abc import ABC, abstractmethod
import jax.numpy as jnp

class Metric(ABC):
  """
  Base class for all Metrics
  
  Metrics are different from Losses in that they are used to evaluate the model, not to train it. Thus NetFlash/StaticNet/NetLab will
  treat an increasing value of a metric as a positive sign.
  
  A metric class must implement the following method
  - '__call__' : a method that contains the logic for computing the metric
    - Args:
      - y_true (jnp.ndarray) : The true class labels, must be the same format as y_pred
      - y_pred (jnp.ndarray) : The predicted values that is the direct output of the sequential class
    - Returns:
      - float : the score of the predicted values
  """
  @abstractmethod
  def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    A metric class must implement the following method
    - '__call__' : a method that contains the logic for computing the metric
      - Args:
        - y_true (jnp.ndarray) : The true class labels, must be the same format as y_pred
        - y_pred (jnp.ndarray) : The predicted values that is the direct output of the sequential class
      - Returns:
        - float : the score of the predicted values
    """
    pass

class Accuracy(Metric):
  @staticmethod
  def __call__(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Accuracy
    -----
      Returns the accuracy of the model.
      Applicable for both binary and multi-class classification.
    -----
    Args
    -----
    - y_true (jnp.ndarray) : The true class labels (e.g., [0, 1, 0] or one-hot encoded [[1,0],[0,1],[1,0]]).
    - y_pred (jnp.ndarray) : The predicted values.
                            For classification: can be logits, probabilities, or hard predictions.
                            If logits/probabilities, argmax will convert to hard predictions.

    Returns
    -----
      (jnp.ndarray) : The proportion of correctly classified samples (0.0 to 100.0). Returns 0.0 for empty input.
    """
    if y_true.size == 0:
      return jnp.array(0.0)

    # Ensure y_true is 1D labels if it's one-hot encoded
    if y_true.ndim > 1 and y_true.shape[-1] > 1:
      y_true_labels = jnp.argmax(y_true, axis=-1)
    else:
      y_true_labels = y_true.astype(jnp.int32) # Ensure integer labels

    # Convert y_pred to hard predictions if it's logits or probabilities
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1: # Multi-class logits/probabilities
      y_pred_labels = jnp.argmax(y_pred, axis=-1)
      
    elif y_pred.ndim == 1 and y_pred.dtype == jnp.float32: # Binary probabilities (e.g., after sigmoid)
      y_pred_labels = (y_pred > 0.5).astype(jnp.int32)
      
    else: # Already hard predictions
      y_pred_labels = y_pred.astype(jnp.int32)

    correct = jnp.sum(y_true_labels == y_pred_labels)
    
    # For classification, total samples is usually the first dimension
    total = y_true.shape[0] if y_true.ndim > 0 else 1 

    return jnp.where(total != 0, correct / total, jnp.array(0.0))

class Precision(Metric):
  @staticmethod
  def __call__(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Precision (for Binary Classification)
    -----
      Returns the precision of the model for the positive class (assumed to be 1).
      Precision = TP / (TP + FP)
      It measures the proportion of positive identifications that were actually correct.
    -----
    Args
    -----
    - y_true (jnp.ndarray) : The true binary labels (0 or 1). Can be one-hot encoded.
    - y_pred (jnp.ndarray) : The predicted values. For binary: probabilities (0-1) or hard predictions (0/1).

    Returns
    -----
      (jnp.ndarray) : The precision score (0.0 to 100.0). Returns 0.0 if (TP + FP) is zero.
    """
    if y_true.size == 0:
      return jnp.array(0.0)

    # Convert y_true to 1D binary labels
    if y_true.ndim > 1 and y_true.shape[-1] > 1: # One-hot encoded y_true
      y_true_binary = jnp.argmax(y_true, axis=-1).astype(jnp.int32)
    else: # Already 1D binary labels
      y_true_binary = y_true.astype(jnp.int32)

    # Convert y_pred to hard binary predictions
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1: # Multi-class output, assume class 1 is positive
      y_pred_binary = (jnp.argmax(y_pred, axis=-1) == 1).astype(jnp.int32)
    elif y_pred.ndim == 1 and y_pred.dtype == jnp.float32: # Probabilities, e.g., from sigmoid
      y_pred_binary = (y_pred > 0.5).astype(jnp.int32)
    else: # Already hard binary predictions
      y_pred_binary = y_pred.astype(jnp.int32)

    true_positive = jnp.sum((y_true_binary == 1) & (y_pred_binary == 1))
    false_positive = jnp.sum((y_true_binary == 0) & (y_pred_binary == 1))

    denominator = true_positive + false_positive
    return jnp.where(denominator != 0, true_positive / denominator, jnp.array(0.0))

class Recall(Metric):
  @staticmethod
  def __call__(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Recall (for Binary Classification)
    -----
      Returns the recall (sensitivity or True Positive Rate) of the model for the positive class (assumed to be 1).
      Recall = TP / (TP + FN)
      It measures the proportion of actual positives that were identified correctly.
    -----
    Args
    -----
    - y_true (jnp.ndarray) : The true binary labels (0 or 1). Can be one-hot encoded.
    - y_pred (jnp.ndarray) : The predicted values. For binary: probabilities (0-1) or hard predictions (0/1).

    Returns
    -----
      (jnp.ndarray) : The recall score (0.0 to 100.0). Returns 0.0 if (TP + FN) is zero.
    """
    if y_true.size == 0:
      return jnp.array(0.0)
    
    # Convert y_true to 1D binary labels
    if y_true.ndim > 1 and y_true.shape[-1] > 1: # One-hot encoded y_true
      y_true_binary = jnp.argmax(y_true, axis=-1).astype(jnp.int32)
    else: # Already 1D binary labels
      y_true_binary = y_true.astype(jnp.int32)

    # Convert y_pred to hard binary predictions
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1: # Multi-class output, assume class 1 is positive
      y_pred_binary = (jnp.argmax(y_pred, axis=-1) == 1).astype(jnp.int32)
    elif y_pred.ndim == 1 and y_pred.dtype == jnp.float32: # Probabilities, e.g., from sigmoid
      y_pred_binary = (y_pred > 0.5).astype(jnp.int32)
    else: # Already hard binary predictions
      y_pred_binary = y_pred.astype(jnp.int32)

    true_positive = jnp.sum((y_true_binary == 1) & (y_pred_binary == 1))
    false_negative = jnp.sum((y_true_binary == 1) & (y_pred_binary == 0))

    denominator = true_positive + false_negative
    return jnp.where(denominator != 0, true_positive / denominator, jnp.array(0.0))

class F1_score(Metric):
  @staticmethod
  def __call__(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    F1 Score (for Binary Classification)
    -----
      Returns the F1 score, which is the harmonic mean of Precision and Recall.
      F1 = 2 * (Precision * Recall) / (Precision + Recall)
      It's a balanced measure that considers both precision and recall.
    -----
    Args
    -----
    - y_true (jnp.ndarray) : The true binary labels (0 or 1). Can be one-hot encoded.
    - y_pred (jnp.ndarray) : The predicted values. For binary: probabilities (0-1) or hard predictions (0/1).

    Returns
    -----
      (jnp.ndarray) : The F1 score (0.0 to 1.0). Returns 0.0 if (Precision + Recall) is zero.
    """
    # Note: Precision and Recall functions return 0-100.0, so divide by 100.0 here for F1 calc
    precision_val = Precision(y_true, y_pred) / 100.0
    recall_val = Recall(y_true, y_pred) / 100.0

    denominator = precision_val + recall_val
    return jnp.where(denominator != 0, 2 * (precision_val * recall_val) / denominator, jnp.array(0.0))

class ROC_AUC(Metric):
  @staticmethod
  def __call__(y_true: jnp.ndarray, y_pred_prob: jnp.ndarray) -> jnp.ndarray:
    """
    ROC AUC (Area Under the Receiver Operating Characteristic Curve)
    -----
      Returns the ROC AUC score for binary classification.
      Requires predicted probabilities for the positive class.
      Measures the model's ability to distinguish between classes across various thresholds.
    -----
    Args
    -----
    - y_true (jnp.ndarray) : The true binary labels (0 or 1). Can be one-hot encoded.
    - y_pred_prob (jnp.ndarray) : The predicted probabilities for the positive class (between 0 and 1).

    Returns
    -----
      (jnp.ndarray) : The ROC AUC score (0.0 to 1.0). Returns 0.5 if all predictions are the same or not enough positives/negatives.
    """
    if y_true.size == 0 or y_true.shape[0] != y_pred_prob.shape[0]:
      return jnp.array(0.5) # Return 0.5 for empty or mismatched input, similar to original

    # Ensure y_true is 1D binary labels
    if y_true.ndim > 1 and y_true.shape[-1] > 1: # One-hot encoded y_true
      y_true_binary = jnp.argmax(y_true, axis=-1).astype(jnp.int32)
    else: # Already 1D binary labels
      y_true_binary = y_true.astype(jnp.int32)

    # Ensure y_pred_prob is 1D probabilities for the positive class
    if y_pred_prob.ndim > 1:
      # If y_pred_prob is (N, 2) for binary classification, assume class 1 is positive
      if y_pred_prob.shape[-1] == 2:
        y_pred_prob_pos = y_pred_prob[:, 1]
      else:
        # If it's multi-class, this AUC doesn't directly apply or needs OVR strategy.
        # For simplicity, if it's (N, C > 2), this function might need rethinking or
        # assuming y_pred_prob is already pre-processed to be a 1D probability for positive class
        # (This raise will break JIT, but indicates a conceptual input mismatch)
        # For JIT, you'd handle this with more generic logic or strict input types.
        return jnp.array(0.5) # Fallback to a default if input is not as expected for binary AUC
    else:
      y_pred_prob_pos = y_pred_prob

    # Filter out samples where y_true_binary is not 0 or 1.
    # Use boolean indexing, but ensure the resulting arrays are not empty
    # as that can cause issues with subsequent operations.
    valid_indices = (y_true_binary == 0) | (y_true_binary == 1)
    
    # Use jnp.where for conditional assignment to avoid changing array size dynamically
    y_true_binary_filtered = jnp.where(valid_indices, y_true_binary, -1) # -1 as invalid marker
    y_pred_prob_pos_filtered = jnp.where(valid_indices, y_pred_prob_pos, -1.0)

    n_pos = jnp.sum(y_true_binary_filtered == 1)
    n_neg = jnp.sum(y_true_binary_filtered == 0)

    if n_pos == 0 or n_neg == 0:
      return jnp.array(0.5)

    # Create a permutation that sorts by prediction probability in descending order
    sorted_indices = jnp.argsort(-y_pred_prob_pos_filtered) # Argsort in descending order

    sorted_y_true = y_true_binary_filtered[sorted_indices]
    # sorted_y_pred_prob is implicitly sorted by this index.

    # Calculate cumulative sums of true positives and false positives
    tps = jnp.cumsum(sorted_y_true == 1)
    fps = jnp.cumsum(sorted_y_true == 0)

    # Calculate TPR and FPR, adding (0,0) and (1,1) points explicitly for robustness
    # Concatenate arrays for the full ROC curve points
    fpr = jnp.concatenate([jnp.array([0.0]), fps / n_neg, jnp.array([1.0])])
    tpr = jnp.concatenate([jnp.array([0.0]), tps / n_pos, jnp.array([1.0])])

    # Remove duplicate (FPR, TPR) points to ensure correct area calculation
    # This is a bit tricky with JAX's static shapes. A common way is to compute
    # differences and filter. For simplicity and general robustness, we can
    # assume slight numerical differences will be handled by trapezoidal rule.
    # For exact duplicate removal in JAX, one usually needs more advanced techniques
    # or acceptance of slight precision differences.
    
    # Calculate area using the trapezoidal rule directly
    auc = jnp.sum(jnp.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2.0)
    
    auc = jnp.clip(auc, 0.0, 1.0) # Ensure AUC is within [0, 1] range

    return auc

class R2_Score(Metric):
  @staticmethod
  def __call__(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    R2 Score (Coefficient of Determination)
    -----
      Returns the R2 score for regression tasks.
      It indicates the proportion of the variance in the dependent variable
      that is predictable from the independent variables.
      R2 = 1 - (SS_res / SS_tot)
    -----
    Args
    -----
    - y_true (jnp.ndarray) : The true numerical values.
    - y_pred (jnp.ndarray) : The predicted numerical values.

    Returns
    -----
      (jnp.ndarray) : The R2 score. Returns 0.0 if total sum of squares is zero.
    """
    if y_true.size == 0 or y_true.shape[0] != y_pred.shape[0]:
      return jnp.array(0.0)
      
    mean_y_true = jnp.mean(y_true) 

    ss_total = jnp.sum(jnp.square(y_true - mean_y_true))
    ss_residual = jnp.sum(jnp.square(y_true - y_pred))

    return jnp.where(
      ss_total == 0,
      # If ss_total is 0:
      #   If ss_residual is 0: return 0.0 (perfect fit for a constant, as per original's 0.0)
      #   If ss_residual is not 0: return -1e+10 (very bad fit for a constant, as per original)
      jnp.where(ss_residual == 0, jnp.array(0.0), jnp.array(-1e+10)), 
      1 - (ss_residual / ss_total)
    )
    
  