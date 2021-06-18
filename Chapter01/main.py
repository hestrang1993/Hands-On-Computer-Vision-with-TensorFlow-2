"""
The :mod:`Chapter01.main` module will run my simple network and plot the results using matplotlib.
"""
import mnist
import numpy as np

random_seed = 42
"""
int: The seed value for the random number generator (RNG).
"""

x_train = mnist.train_images()
"""
ndarray: The training images from Yann LeCun MNIST database as a numpy array.
"""

y_train = mnist.train_labels()
"""
ndarray: The training labels from Yann LeCun MNIST database as a numpy array.
"""

x_test = mnist.test_images()
"""
ndarray: The test images from Yann LeCun MNIST database as a numpy array.
"""

y_test = mnist.test_labels()
"""
ndarray: The test labels from Yann LeCun MNIST database as a numpy array.
"""


def format_dataset(dataset_ndarray):
    """
    Format the MNIST datasets into normalized 1 by 784 numpy arrays.

    Parameters
    ----------
    dataset_ndarray : ndarray
        The training and testing images.

    Returns
    -------
    ndarray
        The formatted MNIST training and testing images.
    """
    dataset_ndarray_reshape = dataset_ndarray.reshape(-1, 28 * 28)
    dataset_ndarray_reshape_normalize = dataset_ndarray_reshape / 255.0
    return dataset_ndarray_reshape_normalize


def one_hot_dataset(dataset_ndarray):
    number_of_classes = 10
    dataset_ndarray_one_hot = np.eye(number_of_classes)[dataset_ndarray]
