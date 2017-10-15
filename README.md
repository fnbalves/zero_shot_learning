1) Get CIFAR-10 and CIFAR-100 from:

https://www.cs.toronto.edu/~kriz/cifar.html

Put cifar-100 test, train and meta files inside a folder called cifar-100-python/
(read_cifar100 conde uses them)

2) Get pre-trained word2vec from:

https://github.com/idio/wiki2vec

Put en.model file inside a folder called en_1000_no_stem
(If you wish to create the mongo database, use a computer with more than 10Gb RAM, after that, you can
make a dump of the database and load it in a simpler computer) 

3) If you wish to use smaller glove models, download one from the list in:

https://github.com/3Top/word2vec-api

After that, modify the glove_interface.py file inserting the right glove_data_file

4) Install python dependencies by using pip and the requirements.txt file:

sudo pip install requirements.txt