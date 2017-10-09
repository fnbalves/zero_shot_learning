import pickle
import random

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
    
    for i in range(len_data):
        new_entry = [cifar_dict['data'][i], cifar_dict['fine_labels'][i],
                     cifar_dict['coarse_labels'][i]]
    
        coarse = cifar_dict['coarse_labels'][i]
        fine = cifar_dict['fine_labels'][i]

        table = correspondence_table[coarse]
        
        if fine in table[:3]:
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

    separated_data = separate_target_data(cifar_train_dict, corrs_coarse_fine)

    target_data = separated_data['target']
    not_target_data = separated_data['not_target']

    print('BUILDING DATASET WITH STR LABELS FOR NEW NORMALIZATION')
    str_target_data = create_dataset_with_string_labels(target_data, cifar_meta)
    str_not_target_data = create_dataset_with_string_labels(not_target_data, cifar_meta)

    print('SAVING...')
    out_target = open('pickle_files/target_data.pickle', 'wb')
    out_not_target = open('pickle_files/not_target_data.pickle', 'wb')

    pickle.dump(str_target_data, out_target)
    pickle.dump(str_not_target_data, out_not_target)

    out_target.close()
    out_not_target.close()
    print('DONE!')
