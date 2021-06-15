"""
The :mod:`activation_functions` module will contain several activation functions and their derivatives.

.. topic:: Activation Functions

    This module will contain the following activation functions:

    * The sigmoid function :math:`S\left(y\\right) = \\frac{1}{1 + e^{-y}}`
    * The L2 loss function :math:`L_{2} \left({y_{i}}^{true}, y_{i} \\right) = \sum{}_i \left( {y_{i}}^{true} - y_{i}
    \\right)^2`
    * The binary cross entropy function :math:`\\text{BCE} \left( {y_{i}}^{true} , {y_{i}} \\right ) = \sum_{i}
    \left[ {-y_{i}}^{true} \log \left( y_{i} \\right) + \left( 1 - {y_{i}}^{true} \\right) \log \left( 1 - y_{i}
    \\right ) \\right]`
"""

# ==============================================================================
# Imported Modules
# ==============================================================================

import numpy as np


# ==============================================================================
# Function Definitions
# ==============================================================================

def sigmoid_function(x):
    """
    Use the sigmoid function as an activation function.

    .. math::
        S\left(y\\right) = \\frac{1}{1 + e^{-y}}

    Parameters
    ----------
    x : int or float
        The weighted sum.

    Returns
    -------
    float
        The output of the neuron based on the sigmoid function.
    """
    numerator = 1
    denominator = (1 + np.exp(-x))
    y = numerator / denominator
    return y


def derivative_of_sigmoid_function(x):
    """
    The derivative of the sigmoid function.

    .. math::
        \\frac{ \\mathrm{ dS } }{ \\mathrm{ d } y } = y \left( 1 - y \\right)

    Parameters
    ----------
    x

    Returns
    -------

    """
    y = x * (1 - x)
    return y
