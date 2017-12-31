import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

NUM_CHANNELS = 3


def image_array_to_image_matrix(image_array):
    """Gets a image matrix from CIFAR-100 and turns it into a matrix suitable for CNN Networks"""
    image_size = int(np.sqrt(np.prod(np.shape(image_array)) / NUM_CHANNELS))

    image_matrix = image_array.reshape(NUM_CHANNELS, image_size, image_size).transpose(1, 2, 0)
    return image_matrix


def image_matrix_to_image_array(image_matrix):
    """Gets a image matrix for use in CNNs and converts it back to a CIFAR-100 like matrix"""
    image_size = np.shape(image_matrix)[1]

    image_array = image_matrix.transpose(2, 0, 1).reshape(image_size * image_size * NUM_CHANNELS, )
    return image_array


def visualize_image(image):
    """Displays the image in a CIFAR-100 matrix"""
    num_dims = len(np.shape(image))
    if num_dims == 1:
        image_to_visualize = image_array_to_image_matrix(image)
    elif num_dims == 3:
        image_to_visualize = image
    else:
        raise ValueError("Image array should have one or three dimensions")

    plt.figure()
    plt.imshow(image_to_visualize, interpolation='nearest')
    plt.show()


def resize_image_matrix(image_matrix, new_x_size, new_y_size):
    """Resize a image matrix"""
    new_image = scipy.misc.imresize(image_matrix, (new_x_size, new_y_size))
    return new_image
