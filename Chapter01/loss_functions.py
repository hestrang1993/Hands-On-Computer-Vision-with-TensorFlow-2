"""
The :mod:`loss_functions` module will contain several loss functions and their derivatives.
"""

# ==============================================================================
# Imported Modules
# ==============================================================================

import numpy as np


# ==============================================================================
# Loss Functions
# ==============================================================================

def l2_loss_function(y_predicted, y_true):
    """
    Use the L2 loss function to calculate the loss for a model.

    .. math::
        L_{2} \\left( y , y^{true} \\right) = \\sum_{i} { \\left( y_{i}^{true} - y_{i} \\right) }^2

    Parameters
    ----------
    y_predicted : ndarray
        The predicted output of the model.
    y_true : ndarray
        The actual output value.

    Returns
    -------
    int or float
        The value of the L2 loss function.
    """
    batch_size = y_predicted.shape[0]
    square_difference = np.square(y_true - y_predicted)
    sum_of_squares = np.sum(square_difference)
    y = sum_of_squares / batch_size
    return y


def l1_loss_function(y_predicted, y_true):
    """
    Use the L1 loss function to calculate the loss for a model.

    .. math::
        L_{1} \\left( y , y^{true} \\right) = \\sum_{i} \\left| ( y_{i}^{true} - y_{i} \\right|

    Parameters
    ----------
    y_predicted : ndarray
        The predicted output of the model.
    y_true : ndarray
        The actual output value.

    Returns
    -------
    int or float
        The value of the L1 loss function.
    """
    batch_size = y_predicted.shape[0]
    absolute_difference = np.abs(y_true - y_predicted)
    sum_of_absolutes = np.sum(absolute_difference)
    y = sum_of_absolutes / batch_size
    return y


def binary_cross_entropy_loss_function(y_predicted, y_true):
    """
    Use the binary cross entropy (BCE) loss function for a model.

    .. math::
        BCE \\left( y_{i}^{true} , y_{i} \\right) = \\sum_{i} \\left( -y_{i}^{true} \\log \\left( y_{i} \\right) +
        \\left( 1 - y_{i}^{true} \\right) \\log \\left( 1 - y_{i} \\right) \\right)

    Parameters
    ----------
    y_predicted : ndarray
        The predicted output of the model.
    y_true : ndarray
        The actual output value.

    Returns
    -------
    int or float
        The value of the L1 loss function.
    """
    a = np.multiply(y_true, np.log(y_predicted))
    b = np.multiply(np.log(1 - y_predicted), (1 - y_true))
    y = -np.mean(a + b)
    return y


# ==============================================================================
# Derivatives of the Loss Functions
# ==============================================================================

def derivative_of_l2_loss_function(y_predicted, y_true):
    """
    Use the derivative of the L2 loss function.

    Parameters
    ----------
    y_predicted : ndarray
        The predicted output of the model.
    y_true : ndarray
        The actual output value.

    Returns
    -------
    int or float
        The derivative of the L2 loss function.
    """
    y = 2 * (y_predicted - y_true)
    return y


def derivative_of_binary_cross_entropy_loss_function(y_predicted, y_true):
    """
    Use the derivative of the binary cross entropy (BCE) loss function.

    Parameters
    ----------
    y_predicted : ndarray
        The predicted output of the model.
    y_true : ndarray
        The actual output value.

    Returns
    -------
    int or float
        The derivative of the BCE loss function.
    """
    numerator = (y_predicted - y_true)
    denominator = (y_predicted * (1 - y_predicted))
    y = numerator / denominator
    return y

# TODO: Add derivative to the L1 loss function.
