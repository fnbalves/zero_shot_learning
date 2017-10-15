from pymongo import MongoClient
import pickle
import numpy as np

print('OPENING CLIENT')
client = MongoClient('localhost', 27017)
print('CLIENT OPENED')

print('LOADING word2id DICTIONARY')
word2id_dict = pickle.load(open('pickle_files/word2id.pickle', 'rb'), encoding='latin1')
print('DICTIONARY OPENED')

db = client['word2vec']
collection = db['words']

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
		object_id = word2id_dict[word]
		query = {'_id' : object_id}
		results = collection.find_one(query)
		vect = results['vect']
		return np.array(vect, dtype=np.float32)
	except KeyError:
		return None
