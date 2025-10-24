"""
Defaults
=====

A centralized location for default values of internal variables and/or values
"""

# Activation functions
#   these values are only used when the function is used outside of any PyNet APIs since all layers in this
#   framework will initialize all alpha and beta values regardless of the function type

ELU_alpha_default = 1.0

PReLU_alpha_default = 0.1

SELU_alpha_default = 1.05
SELU_beta_default = 1.05

SiLU_alpha_default = 1.0

# Parametric parameters
#   default alpha and beta values for parametric activation functions when initialized inside any of the layers in this framework

parametric_alpha_default = 1.0
parametric_beta_default = 1.0

# zerodiv prevention

epsilon_default = 1e-10