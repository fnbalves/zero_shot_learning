#Partialy based on https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
#Function to train the basic AlexNet model (for testing only)
import tensorflow as tf
import numpy as np
import pickle
import os
from datetime import datetime
from models import AlexNet
from batch_making import *
from training_utils import *

initial_learning_rate = 0.1
momentum = 0.9
num_epochs = 300
batch_size = 128

dropout_rate = 0.5
num_classes = 60
train_layers = ['conv1', 'conv2', 'fc3', 'fc4', 'fc5']

display_step = 1

filewriter_path = 'cifar100_history/'
checkpoint_path = 'checkpoints/'

IMAGE_SIZE = 24
OUTPUT_FILE_NAME = 'train_output.txt'

decay_steps = int(len(target_train_data)/batch_size)
learning_rate_decay_factor = 0.95

if not os.path.isdir(filewriter_path): os.mkdir(filewriter_path)
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

x = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
y = tf.placeholder(tf.float32, [None, num_classes])

model = AlexNet(x, num_classes)
score = model.fc5

var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

initial_x_batch = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
dist_x_batch = distorted_batch(initial_x_batch, IMAGE_SIZE)

with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                            logits = score, labels = y))

with tf.name_scope('train'):
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))
    global_step = tf.Variable(0)
    
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                  global_step,
                                  decay_steps,
                                  learning_rate_decay_factor,
                                  staircase=True)

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    in_k = tf.cast(tf.nn.in_top_k(score, tf.argmax(y,1), 3), tf.float32)
    accuracy_k = tf.reduce_mean(in_k)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = get_batches(target_train_data, batch_size, IMAGE_SIZE)
val_generator = get_batches(target_test_data, batch_size, IMAGE_SIZE)

# Start Tensorflow session
with tf.Session() as sess:

  # Initialize all variables
  sess.run(tf.global_variables_initializer())

  # Load the pretrained weights into the non-trainable layer
  #saver.restore(sess, 'checkpoints_old2/model_epoch42.ckpt')

  print_in_file("{} Start training...".format(datetime.now()))
  print_in_file("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                    filewriter_path))

  # Loop over number of epochs
  for epoch in range(num_epochs):

    print_in_file("{} Epoch number: {}".format(datetime.now(), epoch+1))

    for batch_xs, batch_ys in train_generator:

        # And run the training op
        new_batch = sess.run(dist_x_batch, feed_dict={initial_x_batch: batch_xs})
        
        sess.run(train_op, feed_dict={x: new_batch,
                                          y: batch_ys})


    # Validate the model on the entire validation set
    print_in_file("{} Start validation".format(datetime.now()))
    test_acc = 0.
    test_acc_k = 0.
    
    test_count = 0
    for batch_tx, batch_ty in val_generator:
        acc = sess.run(accuracy, feed_dict={x: batch_tx,
                                                y: batch_ty})
        acc_k = sess.run(accuracy_k, feed_dict={x: batch_tx,
                                                y: batch_ty})
        
        test_acc += acc
        test_acc_k += acc_k
        test_count += 1
    test_acc /= test_count
    test_acc_k /= test_count
    
    print_in_file("Validation Accuracy = %s %.4f" % (datetime.now(), test_acc), OUTPUT_FILE_NAME)
    print_in_file("Validation Accuracy (k-top) = %s %.4f" % (datetime.now(), test_acc_k))

    # Reset the file pointer of the image data generator
    train_generator = get_batches(target_train_data, batch_size, IMAGE_SIZE)
    val_generator = get_batches(target_test_data, batch_size, IMAGE_SIZE)

    print_in_file("{} Saving checkpoint of model...".format(datetime.now()), OUTPUT_FILE_NAME)

    #save checkpoint of the model
    checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch)+'.ckpt')
    save_path = saver.save(sess, checkpoint_name)

    print_in_file("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name), OUTPUT_FILE_NAME)
