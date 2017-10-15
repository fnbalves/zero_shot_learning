import pandas as pd
import csv

glove_data_file = 'glove.6B/glove.6B.100d.txt'

print('Loading glove model')
words = pd.read_table(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
print('Loaded')

composite_words = {
        'pine_tree' : 'pine',
        'sweet_pepper' : 'pepper',
        'maple_tree' : 'maple',
        'aquarium_fish' : 'fish',
        'willow_tree' : 'willow',
        'pickup_truck' : 'pickup',
        'palm_tree' : 'palm',
        'lawn_mower' : 'mower',
        'oak_tree' : 'oak'
}

def normalize_label(label):
        if label in composite_words:
                return composite_words[label]
        else:
                return label

def find_word_vec(word):
    try:
        return words.loc[word].as_matrix()
    except:
        return None
