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

not_target_labels = ['baby', 'bear', 'beaver', 'bed', 'beetle', 'bowl', 'bridge',
                     'bus', 'camel', 'can', 'caterpillar', 'clock', 'couch', 'crab',
                     'dolphin', 'forest', 'fox', 'hamster', 'house', 'kangaroo', 'lamp',
                     'lizard', 'man', 'maple', 'mouse', 'mower', 'orange', 'orchid', 'palm',
                     'pear', 'pickup', 'plain', 'poppy', 'porcupine', 'ray', 'spider', 'tank',
                     'tiger', 'trout', 'turtle']


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

def cosine_distance(v1, v2):
      return 1 - np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))

def get_closest_words(vector, zero_shot_only=False):
      all_distances = []
      possible_labels = all_labels
      if zero_shot_only:
            possible_labels = not_target_labels
      
      for label in possible_labels:
            wv = find_word_vec(normalize_label(label))
            all_distances.append([label, np.linalg.norm(vector - wv)])
      sorted_dist = sorted(all_distances, key=lambda x: x[1])
      return [s[0] for s in sorted_dist]

def get_closest_words_cosine(vector, zero_shot_only=False):
      all_distances = []
      possible_labels = all_labels
      if zero_shot_only:
            possible_labels = not_target_labels
      
      for label in possible_labels:
            wv = find_word_vec(normalize_label(label))
            all_distances.append([label, cosine_distance(vector, wv)])
      sorted_dist = sorted(all_distances, key=lambda x: x[1])
      return [s[0] for s in sorted_dist]

def get_results(check_point_file, output_file):
      data_generator = get_batches(all_not_target, batch_size, IMAGE_SIZE, word2vec=True, send_raw_str=True)

      points = {}
      for label in all_labels:
          points[normalize_label(label)] = []

      # Start Tensorflow session
      with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Load the pretrained weights into the non-trainable layer
        saver.restore(sess, check_point_file)
        
        distances = {}
        accuracies = {}
        accuracies_superclass = {}
        
        for batch_x, batch_y, batch_labels in data_generator:
            output = sess.run(model_output, {x: batch_x})
            for i, o in enumerate(output):
                label_vec = batch_y[i]
                new_distance = cosine_distance(label_vec, o) #np.linalg.norm(label_vec - o)
                closest_words = get_closest_words_cosine(o, zero_shot_only=False)[:5]
                correct_label = batch_labels[i]

                most_close = closest_words[0]
                most_close_sp = reverse_dic[normalize_label(most_close)]
                label_sp = reverse_dic[normalize_label(correct_label)]
                
                if correct_label not in distances:
                      distances[correct_label] = [new_distance]
                else:
                      distances[correct_label].append(new_distance)

                if correct_label not in accuracies_superclass:
                      if most_close_sp == label_sp:
                            accuracies_superclass[correct_label] = [1.0, 1.0]
                      else:
                            accuracies_superclass[correct_label] = [0.0, 1.0]
                else:
                      if most_close_sp == label_sp:
                            accuracies_superclass[correct_label][0] = accuracies_superclass[correct_label][0] + 1.0
                      accuracies_superclass[correct_label][1] = accuracies_superclass[correct_label][1] + 1.0
                
                if correct_label not in accuracies:
                      
                      if correct_label in closest_words:
                            accuracies[correct_label] = [1.0, 1.0]
                      else:
                            accuracies[correct_label] = [0.0, 1.0]
                else:
                      if correct_label in closest_words:
                            accuracies[correct_label][0] = accuracies[correct_label][0] + 1.0
                      accuracies[correct_label][1] = accuracies[correct_label][1] + 1.0



      for key in distances.keys():
            distances[key] = np.mean(distances[key])
            accuracies[key] = accuracies[key][0]/accuracies[key][1]
            accuracies_superclass[key] = accuracies_superclass[key][0]/accuracies_superclass[key][1]

      out = open(os.path.join(OUTPUT_FILES_FOLDER, output_file + '.pickle'), 'wb')
      pickle.dump({'distances': distances, 'accuracies': accuracies, 'accuracies_superclass': accuracies_superclass}, out)
      out.close()

      print('OUTPUT DONE')

def show_results(result_dict):
      distances = result_dict['distances']
      accuracies = result_dict['accuracies']
      accuracies_superclass = result_dict['accuracies_superclass']
      
      print('DISTANCES')
      for key in distances.keys():
            print(key, distances[key])

      print('ACCURACIES')
      for key in accuracies.keys():
            print(key, accuracies[key])

      print('ACCURACIES SUPERCLASS')
      for key in accuracies_superclass.keys():
            print(key, accuracies_superclass[key])

      plt.hist(accuracies.values(), bins=[a/100.0 for a in range(0, 100, 5)])
      plt.xticks([a/100.0 for a in range(0, 100, 5)])
      plt.yticks(list(range(0,100, 5)))
      plt.show()

if AUTO_COMPUTE:
      for i, check_point_file in enumerate(CHECK_POINT_FILES):
            output_file = OUTPUT_FILES[i]
            print('COMPUTING', check_point_file)
            get_results(check_point_file, output_file)

