"""
# TODO: Write doctring for this module. Finish MyFullyConnectedLayer.
"""

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
        hidden_layers_sizes : (int, int)
            A list of sizes for each hidden layer to add to the network.
        activation_functions : function
            The activation function for all the layers.
        derivative_of_the_activation_function : function
            The derivative of the activation function.
        loss_functions : function
            The loss function to train this network with.
        derivative_of_the_loss_function : function
            The derivative of the loss function, for back-propagation
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
        self.layers = [
                MyFullyConnectedLayer(
                        number_of_inputs = self.layer_sizes[i],
                        layer_size = self.layer_sizes[i + 1],
                        activation_function = self.activation_functions,
                        derivative_of_the_activation_function = self.derivative_of_the_activation_function
                )
                for i in range(len(self.layer_sizes) - 1)
        ]
        self.loss_functions = loss_functions
        self.derivative_of_the_loss_function = derivative_of_the_loss_function
