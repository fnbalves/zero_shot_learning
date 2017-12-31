This project's objective is to reproduce the results from

https://static.googleusercontent.com/media/research.google.com/pt-BR//pubs/archive/41473.pdf

However, we use a smaller dataset (CIFAR-100) with lower resolution images.
In order to do so, new cost functions needed to be created.

To use this code

1) Get CIFAR-10 and CIFAR-100 from:

https://www.cs.toronto.edu/~kriz/cifar.html

Put cifar-100 test, train and meta files inside a folder called cifar-100-python/
(read_cifar100 conde uses them)

2) Download the glove model from

http://nlp.stanford.edu/data/glove.6B.zip

And extract the files into a folder called glove.6B

3) Install python dependencies by using pip and the requirements.txt file:

sudo pip install requirements.txt

4) Create a folder called pickle files and run read_cifar100 to create all datasets

5) To train the composite model, run the train_composite file

6) To visualize the TSNE plots, run the visualize_results file (Change the indicated vars on the code)

7) To compute quantitative results, run the compute_quantitative_results file and use the functions
(Change the indicated vars on the code)