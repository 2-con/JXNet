"""
Configuration
=====

This file contains the configuration for the all of the PyNET files.
"""

# classification (low impact)

static_rectifiers                   = ('relu','softplus','mish','swish','leaky relu','gelu','reeu','none','retanh')
parametric_rectifiers               = ('elu', 'selu', 'prelu', 'silu')
normalization_functions             = ('binary step','softsign','sigmoid','tanh')
parametric_normalization_functions  = ()

parametrics = parametric_rectifiers + parametric_normalization_functions
statics = static_rectifiers + normalization_functions

rectifiers = static_rectifiers + parametric_rectifiers
normalization = normalization_functions + parametric_normalization_functions

# skip training if there is less change (high impact)
skip_threshold = 2 # this is different from what is logged.

