"""
# API
Provides 2 main APIs supported by JXNet, each with its own function and structure.

### StandardNet API
  A flexible sequential model for building and running JAX models of neural networks. Inputs an unbatched dataset for training but accepts bulk inference 
  during testing/deployment.
  
### NetLab API
  A utility API with tools to handle the testing and evaluation of multiple JXNet models. Testing and experimentation is at the center
  of the purpose.
"""