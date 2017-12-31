This project objective is to reproduce the results from
https://static.googleusercontent.com/media/research.google.com/pt-BR//pubs/archive/41473.pdf
However, we use a smaller dataset (CIFAR-100) with lower resolution images.
In order to do so, new cost functions needed to be created.

To use this code

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

5) Create a folder called pickle files and run read_cifar100 to create all datasets

6) To train the composite model, run the train_composite file

7) To visualize the TSNE plots, run the visualize_results file (Change the indicated vars on the code)

8) To compute quantitative results, run the compute_quantitative_results file and use the functions
(Change the indicated vars on the code)