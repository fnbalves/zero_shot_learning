import numpy as np
import pickle
from batch_making import *

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


all_labels = pickle.load(open('pickle_files/all_labels.pickle', 'rb'))


def cosine_distance(v1, v2):
    """Computes the cossine distance between two vectors"""
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_closest_words(vector, zero_shot_only=False):
    """Returns the closest words to a vector in a crescent distance order.
    Uses euclidean distance"""
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
    """Returns the closest words to a vector in a crescent distance order.
    Uses cossine distance"""
    all_distances = []
    possible_labels = all_labels
    if zero_shot_only:
        possible_labels = not_target_labels

    for label in possible_labels:
        wv = find_word_vec(normalize_label(label))
        all_distances.append([label, cosine_distance(vector, wv)])
    sorted_dist = sorted(all_distances, key=lambda x: x[1])
    return [s[0] for s in sorted_dist]
