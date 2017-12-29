#Partialy based on https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

import tensorflow as tf
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from datetime import datetime
from models import Composite_model
from batch_making import *
from sklearn.manifold import TSNE

batch_size = 128
num_classes = 60
word2vec_size = 200

IMAGE_SIZE = 24

all_labels = pickle.load(open('pickle_files/all_labels.pickle', 'rb'))

x = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
y = tf.placeholder(tf.float32, [None, word2vec_size])

model = Composite_model(x, num_classes, word2vec_size)
model_output = model.projection_layer

saver = tf.train.Saver()
test_generator = get_batches(target_test_data, batch_size, IMAGE_SIZE, word2vec=True, send_raw_str=True)

def build_all_labels_repr():
      all_repr = []
      for label in all_labels:
            wv = find_word_vec(normalize_label(label))
            all_repr.append(wv)
      return tf.constant(np.array(all_repr), shape=[len(all_labels), word2vec_size], dtype=tf.float32)

points = {}
for label in all_labels:
    points[normalize_label(label)] = []

# Start Tensorflow session
with tf.Session() as sess:

  # Initialize all variables
  sess.run(tf.global_variables_initializer())

  # Load the pretrained weights into the non-trainable layer
  saver.restore(sess, 'checkpoints_devise/model_epoch5.ckpt')


  for batch_x, batch_y, batch_labels in test_generator:
      output = sess.run(model_output, {x: batch_x})
      for i, o in enumerate(output):
          label = normalize_label(batch_labels[i])
          points[label].append(o)

all_points = []
points_labels = []

label_points = [find_word_vec(normalize_label(L)) for L in all_labels]

for k in points.keys():
    for p in points[k]:
        all_points.append(p)
        points_labels.append(k)

for i, L in enumerate(all_labels):
    all_points.append(label_points[i])
    points_labels.append('LABEL-'+normalize_label(L))

print('RUNNING TSNE')
manifold = TSNE(n_components=2).fit_transform(all_points)
print('DONE')

output_points = [a for i, a in enumerate(manifold) if 'LABEL' not in points_labels[i]]
class_points = [[a, points_labels[i]] for i, a in enumerate(manifold) if 'LABEL' in points_labels[i]]

x_output = [a[0] for a in output_points]
y_output = [a[1] for a in output_points]
x_class_points = [a[0][0] for a in class_points]
y_class_points = [a[0][1] for a in class_points]
l_class_points = [a[1].split('-')[1] for a in class_points]

def make_graph(attention_label):
      my_graph = plt.scatter(x_class_points, y_class_points)
      target_point = None
      for i, L in enumerate(l_class_points):
            if L == attention_label:
                target_point = [[x_class_points[i]], [y_class_points[i]]]
            plt.annotate(L, xy=(x_class_points[i], y_class_points[i]), size=5)
      return target_point

def show_label_points(label):
      target_point = make_graph(label)
      wx = [x for i, x in enumerate(x_output) if points_labels[i] == label]
      wy = [y for i, y in enumerate(y_output) if points_labels[i] == label]
      plt.scatter(wx, wy, c='red')
      plt.scatter(target_point[0], target_point[1], c='green')
      plt.title(label)
      plt.savefig(os.path.join('relatorio_cross_ent', label + '.png'), format='png', dpi=1000)

ls = list(set([label for label in points_labels if 'LABEL-' not in label]))
for label in ls:
      plt.close()
      show_label_points(label)


