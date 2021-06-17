"""
TODO: Write introduction for this module later.
"""

import numpy as np


class MyFullyConnectedLayer:
    """
    The :class:`MyFullyConnectedLayer` is a simple fully-connected neural network layer.
    """

    def __init__(
            self,
            number_of_inputs,
            layer_size,
            activation_function,
            derivative_of_the_activation_function = None
    ):
        """
        Create a new :class:`MyFullyConnectedLayer` instance.

        Parameters
        ----------
        number_of_inputs : int
            The input vector size / number of input values.
        layer_size : int
            The output vector size / number of neurons in the layer.
        activation_function : function
            The activation function for this layer.
        derivative_of_the_activation_function : function, optional
            The derivative of the activation function.
        """
        self._weights = np.random.standard_normal((number_of_inputs, layer_size))
        self._biases = np.random.standard_normal((layer_size))
        self._size = layer_size
        self._activation_function = activation_function
        self._derivative_of_the_activation_function = derivative_of_the_activation_function
        self._input = None
        self._output = None
        self._d_loss_d_weights = None
        self._d_loss_d_biases = None

    @property
    def weights(self):
        """
        ndarray: The weight values for each input.
        """
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        self._weights = new_weights

    @property
    def biases(self):
        """
        ndarray: The bias value to add to each weighted sum.
        """
        return self._biases

    @biases.setter
    def biases(self, new_biases):
        self._biases = new_biases

    @property
    def size(self):
        """
        int: The layer size / number of neurons.
        """
        return self._size

    @property
    def activation_function(self):
        """
        function: The activation function computing the neuron's output.
        """
        return self._activation_function

    @property
    def derivative_of_the_activation_function(self):
        """
        function: The derivative of the activation function.

        This function will be used for backpropagation.
        """
        return self._derivative_of_the_activation_function

    @property
    def input(self):
        """
        ndarray: The last provided input vector.

        This will be stored for backpropagation.
        """
        return self._input

    @input.setter
    def input(self, input_in):
        self._input = input_in

    @property
    def output(self):
        """
        ndarray: The corresponding output vector.

        This will be stored for backpropagation.
        """
        return self._output

    @output.setter
    def output(self, output_in):
        self._output = output_in

    @property
    def dl_dw(self):
        """
        ndarray: The derivative of the loss, with respect to the weights.
        """
        return self._d_loss_d_weights

    @dl_dw.setter
    def dl_dw(self, d_l_d_w):
        self._d_loss_d_weights = d_l_d_w

    @property
    def dl_db(self):
        """
        ndarray: The derivative of the loss, with respect to the biases.
        """
        return self._d_loss_d_biases

    @dl_db.setter
    def dl_db(self, d_l_d_b):
        self._d_loss_d_biases = d_l_d_b

    def forward(self, x):
        """
        Forward the input vector through the layer, and return its activation vector.

        .. NOTE::
            Both the argument and return values will be ``ndarray`` of shape ``(batch_size, number_of_inputs)``.

        Parameters
        ----------
        x : ndarray
            The input vector.

        Returns
        -------
        ndarray
            The activation value.
        """
        z = np.dot(x, self.weights) + self.biases
        self.output = self.activation_function(z)
        self.input = x
        return self.output

    def _dy_dz(self):
        """
        Calculate the derivative of the output, with respect to the weighted sum.

        Returns
        -------
        ndarray
            The derivative of the output, with respect to the weighted sum.
        """
        dy_dz = self.derivative_of_the_activation_function(self.output)
        return dy_dz

    def _dl_dz(self, dl_dy):
        """
        Calculate the derivative of the loss, with respect to the output.

        Parameters
        ----------
        dl_dy : ndarray
            The derivative of the loss, with respect to the output.

        Returns
        -------
        ndarray
            The derivative of the loss, with respect to the weighted sum.
        """
        dy_dz = self._dy_dz()
        dl_dz = (dl_dy * dy_dz)
        return dl_dz

    def _dz_dw(self):
        """
        Calculate the derivative of the weighted sum, with respect to the weights.

        Returns
        -------
        ndarray
            The derivative of the weighted sum, with respect to the weights.
        """
        dz_dw = self.input.T
        return dz_dw

    def _dz_dx(self):
        """
        Calculate the derivative of the weighted sum, with respect to the input.

        Returns
        -------
        ndarray
            The derivative of the weighted sum, with respect to the input.
        """
        dz_dx = self.weights.T
        return dz_dx

    def _dz_db(self, dl_dy):
        """
        Calculate the derivative of the weighted sum, with respect to the biases.

        Parameters
        ----------
        dl_dy : ndarray
            The derivative of the loss, with respect to the output.

        Returns
        -------
        ndarray
            The derivative of the weighted sum, with respect to the biases.
        """
        dz_db = np.ones(dl_dy.shape[0])
        return dz_db

    def _calculate_dl_dw(self, dl_dy):
        """
        Calculate the derivative of the loss, with respect to the weights.

        Parameters
        ----------
        dl_dy : ndarray
            The derivative of the loss, with respect to the output.

        Returns
        -------
        None
        """
        dz_dw = self._dz_dw()
        dl_dz = self._dl_dz(dl_dy)
        self.dl_dw = np.dot(dz_dw, dl_dz)

    def _calculate_dl_db(self, dl_dy):
        """
        Calculate the derivative of the loss, with respect to the biases.

        Parameters
        ----------
        dl_dy : ndarray
            The derivative of the loss, with respect to the output.

        Returns
        -------
        None
        """
        dz_db = self._dz_db(dl_dy)
        dl_dz = self._dl_dz(dl_dy)
        self.dl_db = np.dot(dz_db, dl_dz)

    def _calculate_dl_dx(self, dl_dy):
        """
        Calculate the derivative of the loss, with respect to the inputs.

        Parameters
        ----------
        dl_dy : ndarray
            The derivative of the loss, with respect to the output.

        Returns
        -------
        ndarray
            The derivative of the loss, with respect to the inputs.
        """
        dl_dz = self._dl_dz(dl_dy)
        dl_db = self._calculate_dl_db(dl_dy)
        dl_dx = np.dot(dl_dz, dl_db)
        return dl_dx

    def backward(self, dl_dy):
        """
        Back-propagate the loss, computing all the derivatives, storing those with respect to the layer parameters,
        and returning the loss with respect to its inputs for further propagation.

        Parameters
        ----------
        dl_dy : ndarray
            The derivative of the loss, with respect to the output.

        Returns
        -------
        ndarray
            The derivative of the loss, with respect to the input.
        """
        self._calculate_dl_dw(dl_dy)
        self._calculate_dl_db(dl_dy)
        dl_dx = self._calculate_dl_dx(dl_dy)
        return dl_dx

    def optimize(self, epsilon):
        """
        Optimize the layer's parameters, using the stored derivative values.

        Parameters
        ----------
        epsilon : float
            The learning rate.

        Returns
        -------
        None
        """
        self.weights -= epsilon * self.dl_dw
        self.biases -= epsilon * self.dl_db
