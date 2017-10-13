#Partialy based on https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

import tensorflow as tf
import numpy as np
import pickle
import os
from datetime import datetime
from models import AlexNet
from batch_making import *

learning_rate = 0.1
num_epochs = 30
batch_size = 400

dropout_rate = 0.5
num_classes = 60
train_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']

display_step = 1

filewriter_path = 'cifar100_history/'
checkpoint_path = 'checkpoints/'

IMAGE_SIZE = 227

if not os.path.isdir(filewriter_path): os.mkdir(filewriter_path)
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

x = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

model = AlexNet(x, keep_prob, num_classes, train_layers)
score = model.fc8

var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

OUTPUT_FILE_NAME = 'train_output.txt'

def print_in_file(string):
    output_file = open(OUTPUT_FILE_NAME, 'a')
    output_file.write(string + '\n')
    print(string)
    output_file.close()

with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                            logits = score, labels = y))

with tf.name_scope('train'):
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = get_batches(train_data, batch_size)
val_generator = get_batches(test_data, batch_size)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(size_train / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor((len_data - size_train) / batch_size).astype(np.int16)


# Start Tensorflow session
with tf.Session() as sess:

  # Initialize all variables
  sess.run(tf.global_variables_initializer())

  # Load the pretrained weights into the non-trainable layer
  saver.restore(sess, 'model_epoch0.ckpt')

  print_in_file("{} Start training...".format(datetime.now()))
  print_in_file("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                    filewriter_path))

  # Loop over number of epochs
  for epoch in range(num_epochs):

    print_in_file("{} Epoch number: {}".format(datetime.now(), epoch+1))

    step = 1

    while step < train_batches_per_epoch:

        # Get a batch of images and labels
        batch_xs, batch_ys = train_generator.__next__()

        # And run the training op
        sess.run(train_op, feed_dict={x: batch_xs,
                                          y: batch_ys,
                                          keep_prob: dropout_rate})

        step += 1

    # Validate the model on the entire validation set
    print_in_file("{} Start validation".format(datetime.now()))
    test_acc = 0.
    test_count = 0
    for _ in range(val_batches_per_epoch):
        batch_tx, batch_ty = val_generator.__next__()
        acc = sess.run(accuracy, feed_dict={x: batch_tx,
                                                y: batch_ty,
                                                keep_prob: 1.})
        test_acc += acc
        test_count += 1
    test_acc /= test_count
    print_in_file("Validation Accuracy = %s %.4f" % (datetime.now(), test_acc))

    # Reset the file pointer of the image data generator
    train_generator = get_batches(train_data, batch_size)
    val_generator = get_batches(test_data, batch_size)

    print_in_file("{} Saving checkpoint of model...".format(datetime.now()))

    #save checkpoint of the model
    checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch)+'.ckpt')
    save_path = saver.save(sess, checkpoint_name)

    print_in_file("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
