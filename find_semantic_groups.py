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
CHECK_POINT_FILES = [] #Change here
OUTPUT_FILES = [] #Change here
OUTPUT_FILES_FOLDER = '' #Change here

if len(CHECK_POINT_FILES) == 0 or len(OUTPUT_FILES) == 0 or OUTPUT_FILES_FOLDER=='':
    print('Please modify the vars: CHECK_POINT_FILES, OUTPUT_FILES and OUTPUT_FILES_FOLDER')


AUTO_COMPUTE = True

classes = {
'1': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
'2': ['fish', 'ray', 'shark', 'trout'],
'3': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
'4': ['bottle', 'bowl', 'can', 'cup', 'plate'],
'5': ['apple', 'mushroom', 'orange', 'pear', 'pepper'],
'6': ['clock', 'computer', 'keyboard', 'lamp', 'telephone', 'television'],
'7': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
'8': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
'9': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
'10': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
'11': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
'12': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
'13': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
'14': ['crab', 'lobster', 'snail', 'spider', 'worm'],
'15': ['baby', 'boy', 'girl', 'man', 'woman'],
'16': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
'17': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
'18': ['maple', 'oak', 'palm', 'pine', 'willow'],
'19': ['bicycle', 'bus', 'motorcycle', 'pickup', 'train'],
'20': ['mower', 'rocket', 'car', 'tank', 'tractor']
}

reverse_dic = {}

for key in classes.keys():
      for w in classes[key]:
            wv = find_word_vec(w)
            if wv is None:
                  print('Failed', w)
            reverse_dic[w] = key

all_labels = pickle.load(open('pickle_files/all_labels.pickle', 'rb'))

x = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
y = tf.placeholder(tf.float32, [None, word2vec_size])

model = Composite_model(x, num_classes, word2vec_size)
model_output = model.projection_layer

saver = tf.train.Saver()
all_not_target = not_target_train_data + not_target_test_data

def build_all_labels_repr():
      all_repr = []
      for label in all_labels:
            wv = find_word_vec(normalize_label(label))
            all_repr.append(wv)
      return tf.constant(np.array(all_repr), shape=[len(all_labels), word2vec_size], dtype=tf.float32)

def cosine_distance(v1, v2):
      return 1 - np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))

def get_closest_words(vector):
      all_distances = []
      for label in all_labels:
            wv = find_word_vec(normalize_label(label))
            all_distances.append([label, np.linalg.norm(vector - wv)])
      sorted_dist = sorted(all_distances, key=lambda x: x[1])
      return [s[0] for s in sorted_dist]

def get_closest_words_cosine(vector):
      all_distances = []
      for label in all_labels:
            wv = find_word_vec(normalize_label(label))
            all_distances.append([label, cosine_distance(vector, wv)])
      sorted_dist = sorted(all_distances, key=lambda x: x[1])
      return [s[0] for s in sorted_dist]

def get_results(check_point_file):
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
      for i, check_point_file in enumerate(CHECK_POINT_FILES):
            output_file = OUTPUT_FILES[i]
            print('COMPUTING', check_point_file)
            results = get_results(check_point_file)
            out = open(os.path.join(OUTPUT_FILES_FOLDER, output_file + '.pickle'), 'wb')
            pickle.dump(results, out)
            out.close()

print('DONE')
