"""
The :mod:`activation_functions` module will contain several activation functions and their derivatives.
"""

# ==============================================================================
# Imported Modules
# ==============================================================================

import numpy as np

# ==============================================================================
# Activation Functions
# ==============================================================================

def sigmoid_function(x):
    """
    Use the sigmoid function as an activation function.

    .. math::
        S\left(x\\right) = \\frac{1}{1 + e^{-x}}

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
    Use the derivative of the sigmoid function.

    .. math::
        \\frac{ \\mathrm{ dS } }{ \\mathrm{ d } x } = y \left( 1 - x \\right)

    Parameters
    ----------
    x : int or float
        The value back-propagated from the proceeding layer.

    Returns
    -------
    int or float
        The value to back-propagate to the preceding layer.
    """
    y = x * (1 - x)
    return y


def hyperbolic_tangent_function(x):
    """
    Use the hyperbolic tangent function as an activation function.

    .. math::
        \\tanh\left(x\\right) = \\frac{ e^{ x } - e^{ -x } }{ e^{ x } + e^{ -x } }

    .. NOTE::
        This can also be calculated using ``return np.tanh(x)``.


    Parameters
    ----------
    x : int or float
        The weighted sum.

    Returns
    -------
    float
        The output of the neuron based on the hyperbolic tangent function.
    """
    a = np.exp(x)
    b = np.exp(-x)
    numerator = a - b
    denominator = a + b
    y = numerator / denominator
    return y


def derivative_of_hyperbolic_tangent_function(x):
    """
    Use the derivative of the hyperbolic tangent function.

    .. math::
        \\frac{ \\mathrm{ d } }{ \\mathrm{ d } x } \\tanh \\left( x \\right) = 1 - \\tanh^{2} \\left( x \\right)

    Parameters
    ----------
    x : int or float
        The value back-propagated from the proceeding layer.

    Returns
    -------
    int or float
        The value to back-propagate to the preceding layer.
    """
    y = 1 - np.tanh(x) ** 2
    return y


def rectified_linear_unit_function(x):
    """
    Use the rectified linear unit (ReLU) function.

    .. math::
        \\text{ReLU} \\left( x \\right) = \\begin{Bmatrix} 0 & \\text{if } x < 0 \\ x & \\text{if } x \\geq 0 \\end{
        matrix}

    Parameters
    ----------
    x : int or float
        The weighted sum.

    Returns
    -------
    int or float
        The output of the neuron based on the ReLU activation function.
    """
    if x < 0.0:
        y = 0.0
    else:
        y = x
    return y


def derivative_of_rectified_linear_unit_function(x):
    """
    Use the derivative of the rectified linear unit (ReLU) function.

    .. math::
        \\text{ReLU} \\left( x \\right) = \\begin{Bmatrix} 0 & \\text{if } x < 0 \\ 1 & \\text{if } x \\geq 0 \\end{
        matrix}

    Parameters
    ----------
    x : int or float
        The weighted sum.

    Returns
    -------
    int or float
        The output of the neuron based on the ReLU activation function.
    """
    if x < 0:
        y = 0.0
    else:
        y = 1.0
    return y
