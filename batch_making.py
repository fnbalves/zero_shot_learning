from img_util import *
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pickle
import math
import random
import tensorflow as tf

random.seed(0)
np.random.seed(0)

NUM_CHANNELS = 3

print('LOADING DATA')
target_train_data = pickle.load(open('pickle_files/target_train_data.pickle', 'rb'))
target_test_data = pickle.load(open('pickle_files/target_test_data.pickle', 'rb'))

vectorizer = pickle.load(open('pickle_files/vectorizer.pickle', 'rb'))
print('DATA LOADED')

def adjust_data(image_array, image_size):
    image_matrix = image_array_to_image_matrix(image_array)
    big_image = resize_image_matrix(image_matrix, image_size, image_size)
    return big_image

def get_batches(data, size_batch, image_size):
    random.shuffle(target_train_data)
    len_data = len(data)
    num_batches = math.floor(len_data/size_batch)
    for i in range(num_batches):
        new_batch = data[i*size_batch:min(len_data, (i+1)*size_batch)]
        Xs = [adjust_data(b[0], image_size) for b in new_batch]
        
        raw_Ys = [b[1] for b in new_batch]
        Ys = vectorizer.transform(raw_Ys)
        yield [Xs, Ys]
