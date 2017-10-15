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

def separate_target_data(cifar_dict, used_labels):
    #FORMAT: DATA, FINE_LABEL, COARSE_LABEL
    target_data = []
    not_target_data = []
    
    len_data = len(cifar_dict['data'])
    
    for i in range(len_data):
        new_entry = [cifar_dict['data'][i], cifar_dict['fine_labels'][i],
                     cifar_dict['coarse_labels'][i]]

        fine = cifar_dict['fine_labels'][i]
        if fine in used_labels:
            target_data.append(new_entry)
        else:
            not_target_data.append(new_entry)

    return {'target': target_data, 'not_target': not_target_data}

def create_dataset_with_string_labels(dataset, metadata_dic):
    new_dataset = []
    for d in dataset:
        fine_label = d[1]
        coarse_label = d[2]
        fine_name = metadata_dic['fine_label_names'][fine_label]
        coarse_name = metadata_dic['coarse_label_names'][coarse_label]
        new_dataset.append([d[0], fine_name, coarse_name])
    return new_dataset

def build_coarse_to_fine_correspondence(cifar_dict):
    num_coarse = len(set(cifar_dict['coarse_labels']))
    corrs_coarse_fine = []
    for i in range(num_coarse):
        corrs_coarse_fine.append([])
        
    len_dict = len(cifar_dict['data'])

    for i in range(len_dict):
        coarse = cifar_dict['coarse_labels'][i]
        fine = cifar_dict['fine_labels'][i]
        if fine not in corrs_coarse_fine[coarse]:
            corrs_coarse_fine[coarse].append(fine)

    return corrs_coarse_fine

def separated_used_labels(coarse_to_fine_correspondence):
    used_labels = []
    all_labels = []

    for fine_labels in coarse_to_fine_correspondence:
        used_labels = used_labels + fine_labels[:3] #Pick the first three classes for target
        all_labels = all_labels + fine_labels

    return [all_labels, used_labels]

#READING CIFAR 100 DATA

if __name__ == '__main__': 
    cifar_train_dict = read_pickle_file('cifar-100-python/train')
    cifar_test_dict = read_pickle_file('cifar-100-python/test')

    cifar_meta = read_pickle_file('cifar-100-python/meta')
    print('FILES READ')

    print('CALCULATING CORRESPONDENCE')

    corrs_coarse_fine = build_coarse_to_fine_correspondence(cifar_train_dict)
    [all_labels, used_labels] = separated_used_labels(corrs_coarse_fine)
    used_labels_str = [cifar_meta['fine_label_names'][L] for L in used_labels]
    all_labels_str = [cifar_meta['fine_label_names'][L] for L in all_labels]
    print('USED LABELS %d:' % len(used_labels_str), set(used_labels_str))
    print('ALL LABELS %d' % len(all_labels_str), set(all_labels_str))
    
    print('CORRESPONDENCE DONE')

    separated_train_data = separate_target_data(cifar_train_dict, used_labels)
    separated_test_data = separate_target_data(cifar_test_dict, used_labels)
    
    target_train_data = separated_train_data['target']
    target_test_data = separated_test_data['target']
    
    not_target_train_data = separated_train_data['not_target']
    not_target_test_data = separated_test_data['not_target']
    
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
