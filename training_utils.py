import tensorflow as tf
import pickle
from glove_interface import *

all_labels = pickle.load(open('pickle_files/all_labels.pickle', 'rb'))
word2vec_size = 200


def distort_image(image, image_size):
    """Does random distortion at the training images to avoid overfitting"""
    distorted_image = tf.random_crop(image, [image_size, image_size, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)
    float_image = tf.image.per_image_standardization(distorted_image)
    return float_image


def distorted_batch(batch, image_size):
    """Creates a distorted image batch"""
    return tf.map_fn(lambda frame: distort_image(frame, image_size), batch)


def print_in_file(string, output_filename):
    """Prints a string into a file"""
    output_file = open(output_filename, 'a')
    output_file.write(string + '\n')
    print(string)
    output_file.close()


def build_all_labels_repr():
    """Creates a matrix with all labels word2vec representations"""
    all_repr = []
    for label in all_labels:
        wv = find_word_vec(normalize_label(label))
        all_repr.append(wv)
    return tf.constant(np.array(all_repr), shape=[len(all_labels), word2vec_size], dtype=tf.float32)
