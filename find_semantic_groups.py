# Partialy based on https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

import tensorflow as tf
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from datetime import datetime
from models import Composite_model
from batch_making import *
from sklearn.manifold import TSNE
from quantitative_utils import *

batch_size = 128
num_classes = 60
word2vec_size = 200

IMAGE_SIZE = 24
CHECK_POINT_FILES = []  # Change here
OUTPUT_FILES = []  # Change here
OUTPUT_FILES_FOLDER = ''  # Change here

if len(CHECK_POINT_FILES) == 0 or len(OUTPUT_FILES) == 0 or OUTPUT_FILES_FOLDER == '':
    print('Please modify the vars: CHECK_POINT_FILES, OUTPUT_FILES and OUTPUT_FILES_FOLDER')

AUTO_COMPUTE = True

x = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
y = tf.placeholder(tf.float32, [None, word2vec_size])

model = Composite_model(x, num_classes, word2vec_size)
model_output = model.projection_layer

saver = tf.train.Saver()
all_not_target = not_target_train_data + not_target_test_data


def get_results(check_point_file):
    """Create a correlation like matrix between classes from a model checkpoint file"""
    data_generator = get_batches(all_not_target, batch_size, IMAGE_SIZE, word2vec=True, send_raw_str=True)
    all_labels = []
    for k in classes.keys():
        all_labels += [normalize_label(L) for L in classes[k]]
    corr_m = np.zeros((len(all_labels), len(all_labels)))

    points = {}
    for label in all_labels:
        points[normalize_label(label)] = []

    # Start Tensorflow session
    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Load the pretrained weights into the non-trainable layer
        saver.restore(sess, check_point_file)

        for batch_x, batch_y, batch_labels in data_generator:
            output = sess.run(model_output, {x: batch_x})
            for i, o in enumerate(output):
                closest_words = get_closest_words_cosine(o)[:5]
                correct_label = batch_labels[i]
                i1 = all_labels.index(normalize_label(correct_label))
                for cw in closest_words:
                    i2 = all_labels.index(normalize_label(cw))
                    corr_m[i1][i2] += 1
                    corr_m[i2][i1] += 1
        return {'matrix': corr_m, 'labels': all_labels}


def show_results(result):
    """Displays the correlation matrix build by the get_results function"""
    matrix = result['matrix']
    labels = result['labels']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.show()


if AUTO_COMPUTE:
    "Create the correlation matrix from all the checkpoint files"
    for i, check_point_file in enumerate(CHECK_POINT_FILES):
        output_file = OUTPUT_FILES[i]
        print('COMPUTING', check_point_file)
        results = get_results(check_point_file)
        out = open(os.path.join(OUTPUT_FILES_FOLDER, output_file + '.pickle'), 'wb')
        pickle.dump(results, out)
        out.close()

print('DONE')
