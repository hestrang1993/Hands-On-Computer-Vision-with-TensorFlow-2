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

    def backward(self, dl_dy):
        """
        Back-propagate the loss through the layers.

        .. note::
            This function needs ``forward()`` to be called first.

        Parameters
        ----------
        dl_dy : ndarray
            The loss derivative with respect to the network's output.

        Returns
        -------
        ndarray
            The loss derivative with respect to the network's input.
        """
        reversed_layers = reversed(self.layers)
        for layer in reversed_layers:
            dl_dy = layer.backward(dl_dy)
        return dl_dy

    def optimize(self, epsilon):
        """
        Optimize the network parameters according to the stored gradients

        .. note::
            This function needs ``backward()`` to be called first.

        Parameters
        ----------
        epsilon : float
            The learning rate.

        Returns
        -------
        None
        """
        for layer in self.layers:
            layer.optimize(epsilon)

    def _check_prediction(self, x_values, y_values, index):
        """
        Check whether the predicted class from the dataset matches the true class value.

        Parameters
        ----------
        x_values : ndarray
            The dataset generated during a training session.
        y_values : ndarray
            The corresponding ground-truth validation dataset.
        index : int
            The index for the item to look at in both datasets.

        Returns
        -------
        int
            1 if the predicted class matches the ground-truth class. Otherwise, 0.
        """
        predicted_class = self.predict(x_values[index])
        true_class = y_values[index]
        if predicted_class == true_class:
            return 1
        else:
            return 0

    def evaluate_accuracy(self, x_values, y_values):
        """
        Given a dataset and its ground-truth labels, evaluate the current accuracy of the network.

        Parameters
        ----------
        x_values : ndarray
            The input validation dataset.
        y_values : ndarray
            The corresponding ground-truth validation dataset.

        Returns
        -------
        float
            The accuracy of the network.
        """
        number_correct = 0
        for i in range(len(x_values)):
            number_correct += self._check_prediction(x_values, y_values, i)
        return number_correct / len(x_values)

    @staticmethod
    def _get_batches(x_train, y_train, number_of_batches_per_epoch, batch_size = 32):
        """
        Given a dataset and its ground-truth labels, evaluate the current accuracy of the network.

        Parameters
        ----------
        x_train : ndarray
            The input training dataset.
        y_train : ndarray
            The ground-truth dataset for training.
        number_of_batches_per_epoch : int
            The number of batches to get per run-through of the dataset.
        batch_size : int, optional
            The mini-batch size.

        Returns
        -------
        (ndarray, ndarray)
            The training data and ground-truth training labels for a specific epoch, respectively.
        """
        for b in range(number_of_batches_per_epoch):
            batch_index_begin = b * batch_size
            batch_index_end = batch_index_begin + batch_size
            x = x_train[batch_index_begin: batch_index_end]
            y = y_train[batch_index_begin: batch_index_end]
            return x, y

    def _optimize_on_batch(self, x, y, learning_rate = 1e-3):
        """
        Optimize the model after batching the data.

        Parameters
        ----------
        x : ndarray
            The predicted class values.
        y : ndarray
            The ground-truth class values.
        learning_rate : float, optional
            The learning rate to scale the derivatives with.

        Returns
        -------
        float
            The loss value.

            Used to update epoch loss.
        """
        prediction = self.forward(x)
        loss = self.loss_functions(prediction, y)
        dl_dy = self.derivative_of_the_loss_function(prediction, y)
        self.backward(dl_dy)
        self.optimize(learning_rate)
        return loss

    def _log_training_loss_and_validation_accuracy(
            self, epoch_loss, number_of_batches_per_epoch, do_validation,
            x_validation, y_validation, losses_list, accuracies_list, index
            ):
        """
        Log the training loss and validation accuracy.

        This will make it much easier to follow the model's training.

        Parameters
        ----------
        epoch_loss : int or float
            The loss per epoch.
        number_of_batches_per_epoch : int
            The number of batches to get per run-through of the dataset.
        do_validation : bool
            A validation statement to check when I examine the data.
        x_validation : ndarray
            The input validation dataset.
        y_validation : ndarray
            The ground-truth validation dataset.
        losses_list : []
            A list to store the losses in.
        accuracies_list : []
            A list to store the accuracies of each run.
        index : int
            The index for the loop.

        Returns
        -------
        None
        """
        epoch_loss /= number_of_batches_per_epoch
        losses_list.append(epoch_loss)
        if do_validation:
            accuracy = self.evaluate_accuracy(x_validation, y_validation)
            accuracies_list.append(accuracy)
        else:
            accuracy = np.NaN
        print("Epoch {:4d}: training loss = {:.6f} | val accuracy = {:.2f}%".format(index, epoch_loss, accuracy * 100))

    def train(
            self, x_train, y_train, x_validation, y_validation, batch_size = 32, number_of_epochs = 5,
            learning_rate = 1e-3
            ):
        """
        Given a dataset and its ground-truth labels, evaluate the current accuracy of the network.

        Parameters
        ----------
        x_train : ndarray
            The input training dataset.
        y_train : ndarray
            The ground-truth dataset for training.
        x_validation : ndarray, optional
            The input validation dataset.
        y_validation : ndarray, optional
            The ground-truth dataset for validating the model.
        batch_size : int, optional
            The mini-batch size.
        number_of_epochs : int, optional
            The number of training epochs i.e. iterations over the whole dataset.
        learning_rate : float, optional
            The learning rate to scale the derivatives with.

        Returns
        -------
        (list, list)
            The list of training losses for each epoch and the list of validation accuracy values for each epoch,
            respectively.
        """
        number_of_batches_per_epoch = len(x_train) // batch_size
        do_validation = x_validation is not None and y_validation is not None
        losses_list = []
        accuracies_list = []
        for i in range(number_of_epochs):
            epoch_loss = 0
            for j in range(number_of_batches_per_epoch):
                x, y = self._get_batches(x_train, y_train, number_of_batches_per_epoch, batch_size)
                epoch_loss += self._optimize_on_batch(x, y, learning_rate)
            self._log_training_loss_and_validation_accuracy(
                epoch_loss, number_of_batches_per_epoch, do_validation,
                x_validation, y_validation, losses_list, accuracies_list, i
                )
        return losses_list, accuracies_list
