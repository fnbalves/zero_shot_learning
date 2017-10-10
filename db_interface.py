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

def find_word_vec(word):
	try:
		object_id = word2id_dict[word]
		query = {'_id' : object_id}
		results = collection.find_one(query)
		vect = results['vect']
		return np.array(vect, dtype=np.float32)
	except KeyError:
		return None
