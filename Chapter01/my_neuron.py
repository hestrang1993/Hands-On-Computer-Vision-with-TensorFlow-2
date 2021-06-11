"""
The :mod:`Chapter01.my_neuron` module contains the :class:`MyNeuron` class.

The :class:`MyNeuron` class will be the building blocks for the neural network layers I want to create.
"""

# ==============================================================================
# Imported Modules
# ==============================================================================

from numpy import dot
from numpy.random import uniform


# ==============================================================================
# Class Definition
# ==============================================================================

class MyNeuron:
    """
    :class:`MyNeuron` is a digital model of a neuron.
    """

    def __init__(self, number_of_inputs, activation_function):
        """
        Create a new instance of :class:`MyNeuron`.

        Parameters
        ----------
        number_of_inputs : int
            The input vector size or number of input values.
        activation_function : function
            The activation function that will define this neuron.
        """
        # Properties
        self._high_weight_bias = 1.
        self._low_weight_bias = -1.
        self._activation_function = activation_function

        # Attributes
        self.weight = uniform(
                size = number_of_inputs,
                low = self.low_weight_bias,
                high = self.high_weight_bias
        )
        """
        ndarray[float]: The weight value for each input.
        
        The weights :math:`\left( w \\right)` will be a sequence :math:`\left(w_i\\right)_{i=1}^{n}`.
        In this sequence, :math:`w_i` can be any value such that
        
        .. math::
            -1.0 \leq w_i \leq 1.0
        """
        self.bias = uniform(
                size = 1,
                low = self.low_weight_bias,
                high = self.high_weight_bias
        )
        """
        float: The bias value to add to the weighted sum.
        
        The bias :math:`\left( b \\right)` can be any value such that
        
        .. math::
            -1.0 \leq b \leq 1.0
        """

    @property
    def high_weight_bias(self):
        """
        float: The highest value a weight or bias value can take.

        By default, this will be :math:`1.0`.
        """
        return self._high_weight_bias

    @property
    def low_weight_bias(self):
        """
        float: The lowest value a weight or bias value can take.

        By default, this will be :math:`-1.0`.
        """
        return self._low_weight_bias

    @property
    def activation_function(self):
        """
        function: The activation function for this neuron.

        This will be set when :class:`MyNeuron` is instantiated.
        """
        return self._activation_function

    def forward(self, x):
        z = dot(x, self.weight) + self.bias
        return self.activation_function(z)
