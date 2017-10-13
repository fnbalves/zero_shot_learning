#This code reads cifar-100 dataset and picks the first three
#Classes for each superclass to enter the training procedure (target data)
#The other two labels will be used for zero-shot learning.

import pickle
import random
from sklearn.preprocessing import LabelBinarizer

random.seed(0)

def read_pickle_file(filename):
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
    return p

def separate_target_data(cifar_dict, correspondence_table):
    #FORMAT: DATA, FINE_LABEL, COARSE_LABEL
    target_data = []
    not_target_data = []
    
    len_data = len(cifar_dict['data'])
    used_labels = []
    
    for i in range(len_data):
        new_entry = [cifar_dict['data'][i], cifar_dict['fine_labels'][i],
                     cifar_dict['coarse_labels'][i]]
    
        coarse = cifar_dict['coarse_labels'][i]
        fine = cifar_dict['fine_labels'][i]

        table = correspondence_table[coarse]
        used_labels  = used_labels + table[:3]
        
        if fine in table[:3]: #Pick the first three classes for target
            target_data.append(new_entry)
        else:
            not_target_data.append(new_entry)

    return {'target': target_data, 'not_target': not_target_data, 'used_labels': used_labels}

def create_dataset_with_string_labels(dataset, metadata_dic):
    new_dataset = []
    for d in dataset:
        fine_label = d[1]
        coarse_label = d[2]
        fine_name = metadata_dic['fine_label_names'][fine_label]
        coarse_name = metadata_dic['coarse_label_names'][coarse_label]
        new_dataset.append([d[0], fine_name, coarse_name])
    return new_dataset

#READING CIFAR 100 DATA

if __name__ == '__main__': 
    cifar_train_dict = read_pickle_file('cifar-100-python/train')
    cifar_test_dict = read_pickle_file('cifar-100-python/test')

    cifar_meta = read_pickle_file('cifar-100-python/meta')
    print('FILES READ')

    print('CALCULATING CORRESPONDENCE')

    num_coarse = len(set(cifar_train_dict['coarse_labels']))
    corrs_coarse_fine = []
    for i in range(num_coarse):
        corrs_coarse_fine.append([])
        
    len_dict = len(cifar_train_dict['data'])

    for i in range(len_dict):
        coarse = cifar_train_dict['coarse_labels'][i]
        fine = cifar_train_dict['fine_labels'][i]
        if fine not in corrs_coarse_fine[coarse]:
            corrs_coarse_fine[coarse].append(fine)

    print('CORRESPONDENCE DONE')

    separated_train_data = separate_target_data(cifar_train_dict, corrs_coarse_fine)
    separated_test_data = separate_target_data(cifar_test_dict, corrs_coarse_fine)
    
    target_train_data = separated_train_data['target']
    target_test_data = separated_test_data['target']
    
    not_target_train_data = separated_train_data['not_target']
    not_target_test_data = separated_test_data['not_target']
    
    used_labels = separated_train_data['used_labels']
    used_labels_str = [cifar_meta['fine_label_names'][L] for L in used_labels]
    vectorizer = LabelBinarizer()
    vectorizer.fit(used_labels_str)
    
    print('BUILDING DATASET WITH STR LABELS FOR NEW NORMALIZATION')
    str_target_train_data = create_dataset_with_string_labels(target_train_data, cifar_meta)
    str_not_target_train_data = create_dataset_with_string_labels(not_target_train_data, cifar_meta)

    str_target_test_data = create_dataset_with_string_labels(target_test_data, cifar_meta)
    str_not_target_test_data = create_dataset_with_string_labels(not_target_test_data, cifar_meta)
    
    print('SAVING...')
    out_target_train = open('pickle_files/target_train_data.pickle', 'wb')
    out_target_test = open('pickle_files/target_test_data.pickle', 'wb')
    out_not_target_train = open('pickle_files/not_target_train_data.pickle', 'wb')
    out_not_target_test = open('pickle_files/not_target_test_data.pickle', 'wb')
    
    out_vectorizer = open('pickle_files/vectorizer.pickle', 'wb')
    
    pickle.dump(str_target_train_data, out_target_train)
    pickle.dump(str_target_test_data, out_target_test)
    pickle.dump(str_not_target_train_data, out_not_target_train)
    pickle.dump(str_not_target_test_data, out_not_target_test)
    pickle.dump(vectorizer, out_vectorizer)
    
    out_target_train.close()
    out_target_test.close()
    out_not_target_train.close()
    out_not_target_test.close()
    
    out_vectorizer.close()
    print('DONE!')
