from glove_interface import *
used_labels = ['cockroach', 'pine_tree', 'boy', 'trout', 'bed', 'dolphin', 'worm', 'snail', 'butterfly', 'sunflower', 'whale', 'beetle', 'girl', 'apple', 'poppy', 'table', 'forest', 'rabbit', 'bottle', 'sweet_pepper', 'tractor', 'bowl', 'seal', 'chair', 'squirrel', 'maple_tree', 'train', 'tiger', 'skunk', 'cloud', 'mountain', 'bicycle', 'bee', 'otter', 'tank', 'lamp', 'telephone', 'spider', 'aquarium_fish', 'castle', 'mouse', 'ray', 'fox', 'willow_tree', 'raccoon', 'elephant', 'kangaroo', 'pickup_truck', 'orange', 'streetcar', 'palm_tree', 'cattle', 'motorcycle', 'lobster', 'sea', 'man', 'cup', 'couch', 'crab', 'hamster', 'wardrobe', 'snake', 'rose', 'wolf', 'pear', 'lizard', 'plain', 'lawn_mower', 'plate', 'mushroom', 'bear', 'leopard', 'chimpanzee', 'turtle', 'shrew', 'shark', 'tulip', 'orchid', 'can', 'lion', 'flatfish', 'road', 'crocodile', 'woman', 'bus', 'skyscraper', 'oak_tree', 'clock', 'dinosaur', 'possum', 'television', 'house', 'porcupine', 'baby', 'rocket', 'caterpillar', 'beaver', 'camel', 'keyboard', 'bridge']

for label in used_labels:
    wv = find_word_vec(normalize_label(label))
    if wv is None:
        print('FAILED', label)
    else:
        print('SUCCESS', label)
