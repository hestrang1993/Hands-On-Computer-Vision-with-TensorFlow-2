"""
# TODO: Write docstring for this module. Finish MyFullyConnectedLayer.
"""
import numpy as np

from Chapter01.activation_functions import derivative_of_sigmoid_function
from Chapter01.activation_functions import sigmoid_function
from Chapter01.loss_functions import derivative_of_l2_loss_function
from Chapter01.loss_functions import l2_loss_function
from Chapter01.my_fully_connected_layer import MyFullyConnectedLayer


class MySimpleNetwork:
    """
    The :class:`MySimpleNetwork` class is a simple fully connected neural network.
    """

    def __init__(
            self,
            number_of_inputs,
            number_of_outputs,
            hidden_layers_sizes = (64, 32),
            activation_functions = sigmoid_function,
            derivative_of_the_activation_function = derivative_of_sigmoid_function,
            loss_functions = l2_loss_function,
            derivative_of_the_loss_function = derivative_of_l2_loss_function
    ):
        """
        Create a new :class:`MySimpleNetwork` instance.

        Parameters
        ----------
        number_of_inputs : int
            The input vector size / number of input values.
        number_of_outputs : int
            The output vector size.
        hidden_layers_sizes : (int, int), optional
            A list of sizes for each hidden layer to add to the network.
        activation_functions : function, optional
            The activation function for all the layers.
        derivative_of_the_activation_function : function, optional
            The derivative of the activation function.
        loss_functions : function, optional
            The loss function to train this network with.
        derivative_of_the_loss_function : function, optional
            The derivative of the loss function, for back-propagation.
        """
        self.layer_sizes = [
                number_of_inputs,
                *hidden_layers_sizes,
                number_of_outputs
        ]
        """
        list[int, int, int, int]: The dimensions of this simple neural network.
        """
        self.activation_functions = activation_functions
        """
        function: The activation function for all the layers.
        """
        self.derivative_of_the_activation_function = derivative_of_the_activation_function
        """
        function: The derivative of the activation function.
        """
        self.layers = [
                MyFullyConnectedLayer(
                        number_of_inputs = self.layer_sizes[i],
                        layer_size = self.layer_sizes[i + 1],
                        activation_function = self.activation_functions,
                        derivative_of_the_activation_function = self.derivative_of_the_activation_function
                )
                for i in range(len(self.layer_sizes) - 1)
        ]
        """
        list[MyFullyConnectedLayer]: The list of layers forming this simple network.
        """
        self.loss_functions = loss_functions
        """
        function: The loss function to train this network with.
        """
        self.derivative_of_the_loss_function = derivative_of_the_loss_function
        """
        function: The derivative of the loss function, for back-propagation.
        """

    def forward(self, x):
        """
        Forward the input vector through the layers, returning the output vector.

        Parameters
        ----------
        x : ndarray
            The input vector.

        Returns
        -------
        ndarray
            The output activation value.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, x):
        """
        Compute the output corresponding to input ``x``, and return the index of the largest output value.

        Parameters
        ----------
        x : ndarray
            The input vector.

        Returns
        -------
        int
            The predicted class ID.
        """
        estimation = self.forward(x)
        best_class = np.argmax(estimation)
        return best_class
