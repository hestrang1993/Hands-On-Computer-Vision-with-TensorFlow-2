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
