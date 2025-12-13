"""
# Static
A module that contains the necessary components for non-vectorized models to work. Components here are non-negotiable constants and should not be modified
since they are the primary backbone in which non-gradient based JXNet models work.

### Data Fields
  A sub-module containing data fields that are used in non-vectorized models, typically for the classifier and cluster module

### Losses
  A sub-module containing losses that non-vectorized, non-gradient based models accepts as valid loss functions, typically for the classifier and cluster module
"""