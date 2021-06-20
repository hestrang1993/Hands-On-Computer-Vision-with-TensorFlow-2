"""
The :mod:`simple_neural_network_tf_keras` is here to demonstrate how to use TensorFlow to create a simple fully
connected neural network.

I will create a model to analyze the MNIST dataset.
By the end of the day, this model should be able to read hand-written digits with >95% accuracy.

This model was also created to test if the GPU accelerated the training. It did.
"""

import tensorflow as tf

number_of_classes = 10
"""
int: The number of items to classify the dataset items into.
"""
image_rows = 28
"""
int: The number of rows (in pixels) per item in the dataset. 
"""
image_columns = 28
"""
int: The number of columns (in pixels) per item in the dataset. 
"""
number_of_channels = 1
"""
int: The number of color channels in each item of the dataset.
"""
input_shape = (image_rows, image_columns, number_of_channels)
"""
tuple of int, int, int: The shape of each item to test and train on in the dataset.
"""
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


def normalize_x_data(x_data):
    """
    This function will normalize the training and testing images.

    Parameters
    ----------
    x_data : ndarray
        The training and/or testing images.

    Returns
    -------
    numpy.ndarray
        A normalized training and/or testing image values.
    """
    max_value = 255.0
    x_data_normalized = x_data / max_value
    return x_data_normalized


model = tf.keras.models.Sequential()
"""
tensorflow.python.keras.engine.sequential.Sequential: My simple fully connected neural network.

This will be built using TensorFlow.
"""

flattening_layer = tf.keras.layers.Flatten()
"""
tensorflow.python.keras.layers.core.Flatten: A layer to flatten my input data.

I'll add this to my ``model`` instance.
"""

dense_layer_1_units = 128
"""
int: The dimensionality of my processing dense layers.
"""

dense_layer_1_activation = 'relu'
"""
str: The activation function to use for my processing dense layers.
"""

dense_layer_2_activation = 'softmax'
"""
str: The activation function for the last layer in the model.
"""

dense_layer_1 = tf.keras.layers.Dense(units = dense_layer_1_units, activation = dense_layer_1_activation)
"""
tensorflow.python.keras.layers.core.Dense: The processing dense layer of my model.
"""

dense_layer_2 = tf.keras.layers.Dense(units = number_of_classes, activation = dense_layer_2_activation)
"""
tensorflow.python.keras.layers.core.Dense: The final processing dense layer of my model.
"""

model.add(flattening_layer)
model.add(dense_layer_1)
model.add(dense_layer_2)

model_optimizer = 'sgd'
"""
str: A key for my model's optimizer.

Here, I'll use the stochastic gradient descent (SGD) optimizer.
"""

model_loss = 'sparse_categorical_crossentropy'
"""
str: A key for the loss calculation of my model.
"""

model_metrics = ['accuracy']
"""
list[str]: The metric to measure the model on.
"""

model_callbacks = [tf.keras.callbacks.TensorBoard('./keras')]
"""
tensorflow.python.keras.callbacks.TensorBoard: An instance to handle logging the results of the training.
"""

number_of_epochs = 25
"""
int: The number of epochs the model will go through.
"""

model_verbose_key = 1
"""
int: The key for how verbose the model training will be.
"""

x_train = normalize_x_data(x_train)

x_test = normalize_x_data(x_test)

model_validation_data = (x_test, y_test)

if __name__ == '__main__':
    model.compile(optimizer = model_optimizer, loss = model_loss, metrics = model_metrics)
    model.fit(
            x_train, y_train, epochs = number_of_epochs, verbose = model_verbose_key, validation_data =
            model_validation_data, callbacks = model_callbacks
    )
