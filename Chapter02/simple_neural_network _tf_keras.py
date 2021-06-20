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
(x_train, y_train, x_test), (x_test, y_test) = tf.keras.datasets.mnist.load_data()