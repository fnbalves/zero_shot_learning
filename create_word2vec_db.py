from pymongo import MongoClient
from gensim.models import Word2Vec
import pickle

print('LOADING MODEL')
model = Word2Vec.load('en_1000_no_stem/en.model')
print('MODEL LOADED')

vocabulary = model.vocab.keys()
len_vocabulary = len(vocabulary)

client = MongoClient('localhost', 27017)

db = client['word2vec']
collection = db['words']

correspondence_dict = {}

for i, word in enumerate(vocabulary):
	print(i + 1, 'of', len_vocabulary) 
	new_document = {'word': word, 'vect': model.wv[word].tolist()}

	id = collection.insert_one(new_document).inserted_id
	correspondence_dict[word] = id
	print('ID', id)

out = open('pickle_files/word2id.pickle', 'wb')
pickle.dump(correspondence_dict, out)
out.close()

print('OUTPUT FILE SAVED')
