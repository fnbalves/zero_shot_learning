#Partialy based on https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

import tensorflow as tf
import numpy as np
import pickle
import math
import os
from datetime import datetime
from models import Devise
from batch_making import *

initial_learning_rate = 0.01
momentum = 0.9
num_epochs = 300
batch_size = 128

dropout_rate = 0.5
num_classes = 60
word2vec_size = 200

display_step = 1

filewriter_path = 'cifar100_devise_history/'
checkpoint_path = 'checkpoints_devise_cross_ent/'

IMAGE_SIZE = 24
OUTPUT_FILE_NAME = 'train_output_cross_ent.txt'
LOSS_MARGIN = 0.1 #1

decay_steps = int(len(target_train_data)/batch_size)
learning_rate_decay_factor = 0.95

if not os.path.isdir(filewriter_path): os.mkdir(filewriter_path)
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

all_labels = pickle.load(open('pickle_files/all_labels.pickle', 'rb'))

x = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
y = tf.placeholder(tf.float32, [batch_size, word2vec_size])

model = Devise(x, num_classes, word2vec_size)
model_output = model.projection_layer

var_list = [v for v in tf.trainable_variables()]

def distort_image(image):
      distorted_image = tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
      distorted_image = tf.image.random_flip_left_right(distorted_image)
      distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
      distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)
      float_image = tf.image.per_image_standardization(distorted_image)
      return float_image

def distorted_batch(batch):
    return tf.map_fn(lambda frame: distort_image(frame), batch)

initial_x_batch = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
dist_x_batch = distorted_batch(initial_x_batch)

def print_in_file(string):
    output_file = open(OUTPUT_FILE_NAME, 'a')
    output_file.write(string + '\n')
    print(string)
    output_file.close()

def build_all_labels_repr():
      all_repr = []
      for label in all_labels:
            wv = find_word_vec(normalize_label(label))
            all_repr.append(wv)
      return tf.constant(np.array(all_repr), shape=[len(all_labels), word2vec_size], dtype=tf.float32)

def build_relevance_weights(target_labels, R):
      NEG_MARGIN = 1.5
      t_splits = tf.split(target_labels, batch_size, axis=0)
      R_splits = tf.split(R, len(all_labels), axis=0)
      diffs = []
      for t in t_splits:
            new_diff_array = []
            for r in R_splits:
                  new_norm = tf.pow(tf.norm(t - r),2) - NEG_MARGIN
                  new_diff_array.append(new_norm)
            diffs.append(new_diff_array)
      diff_tensor = tf.convert_to_tensor(diffs)
      return diff_tensor

def build_diffs_cross_entropies(model_output, R):
      R_splits = tf.split(R, len(all_labels), axis=0)
      diffs = []

      for r in R_splits:
            r_softmax = tf.nn.softmax(r)
            repeated_r_softmax = tf.reshape(tf.stack([r_softmax]*batch_size), model_output.get_shape())
            cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=model_output,
                                                                      labels = repeated_r_softmax)
            diffs.append(cross_entropies)
      diff_tensor = tf.convert_to_tensor(diffs)
      return diff_tensor

def build_diffs_eucli(model_output, R):
      R_splits = tf.split(R, len(all_labels), axis=0)
      diffs = []

      for r in R_splits:
            repeated_r = tf.reshape(tf.stack([r]*batch_size), model_output.get_shape())
            new_diffs = tf.norm(model_output - repeated_r)
            diffs.append(new_diffs)
      diff_tensor = tf.convert_to_tensor(diffs)
      return diff_tensor

def build_eucli_loss(model_output, target_labels):
      R = build_all_labels_repr()
      proj1 = tf.norm(model_output - target_labels)
      proj2 = (-1)*build_diffs_eucli(model_output, R)
      proj_sum = proj1 + proj2
      proj_mean = tf.reduce_mean(proj_sum)
      final_loss = proj_mean
      return final_loss

def build_cross_ent_loss(model_output, target_labels, use_reg=True):
      R = build_all_labels_repr()
      softmax_target_labels = tf.nn.softmax(target_labels)
      proj1 = tf.nn.softmax_cross_entropy_with_logits(logits=model_output,
                                                     labels=softmax_target_labels)
      proj2 = (-1)*build_diffs_cross_entropies(model_output, R)
      proj_sum = proj1 + proj2
      proj_mean = tf.reduce_mean(proj_sum)
      reg_term = tf.norm(tf.reduce_mean(model_output, 0) - tf.reduce_mean(R, 0))
      reg_relevance = 0.2
      if not use_reg:
            reg_relevance = 0
      final_loss = proj_mean + reg_relevance*reg_term
      return final_loss

def build_prod_loss(model_output, target_labels, use_reg=True):
      R = build_all_labels_repr()
      proj1 = tf.diag_part(tf.matmul(model_output, tf.transpose(target_labels)))
      sum1 = LOSS_MARGIN - proj1
      sum2 = tf.matmul(model_output, tf.transpose(R))
      sum3 = tf.transpose(sum1 + tf.transpose(sum2))
      relu_sum3 = tf.nn.relu(sum3)
      mean = tf.reduce_mean(relu_sum3)
      reg_term = tf.norm(tf.reduce_mean(model_output, 0) - tf.reduce_mean(R, 0))
      reg_relevance = 0.2
      if not use_reg:
            reg_relevance = 0
      
      final_loss = mean + reg_relevance*reg_term
      return final_loss

def build_rel_w_prod_loss(model_output, target_labels, use_reg=True):
      R = build_all_labels_repr()
      proj1 = tf.diag_part(tf.matmul(model_output, tf.transpose(target_labels)))
      sum1 =  LOSS_MARGIN - proj1
      relevance_weights = build_relevance_weights(target_labels, R)
      
      sum2 = tf.matmul(model_output, tf.transpose(R))
      weighted_sum2 = tf.multiply(sum2, relevance_weights)
      sum3 = tf.transpose(sum1 + tf.transpose(weighted_sum2))
      relu_sum3 = tf.nn.relu(sum3)
      mean = tf.reduce_mean(relu_sum3)
      reg_term = tf.norm(tf.reduce_mean(model_output, 0) - tf.reduce_mean(R, 0))
      reg_relevance = 0.2
      if not use_reg:
            reg_relevance = 0
      final_loss = mean + reg_relevance*reg_term
      return final_loss

def build_no_margin_prod_loss(model_output, target_labels):
      R = build_all_labels_repr()
      proj1 = tf.diag_part(tf.matmul(model_output, tf.transpose(target_labels)))
      sum1 =  (-1)*proj1
      sum2 = tf.matmul(model_output, tf.transpose(R))
      sum3 = tf.transpose(sum1 + tf.transpose(sum2))
      mean = tf.reduce_mean(sum3)
      reg_term = tf.norm(tf.reduce_mean(model_output, 0) - tf.reduce_mean(R, 0))
      variance = tf.pow(tf.reduce_mean(tf.norm(model_output, axis=1)) - tf.reduce_mean(tf.norm(R, axis=1)) ,2)
      reg_term2 = variance
      final_loss = mean + 0.2*reg_term + 0.8*reg_term2
      return final_loss

def build_loss(model_output, target_labels):
      return build_cross_ent_loss(model_output, target_labels, use_reg=True)

with tf.name_scope("loss"):
    loss = build_loss(model_output, y)

with tf.name_scope('train'):
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))
    global_step = tf.Variable(0)
    
    learning_rate = initial_learning_rate
    #tf.train.exponential_decay(initial_learning_rate,
    #                              global_step,
    #                              decay_steps,
    #                              learning_rate_decay_factor,
    #                              staircase=True)

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()
variables_to_restore = [v for v in tf.trainable_variables() if 'proj' not in v.name.split('/')[0]]

previous_loader = tf.train.Saver(variables_to_restore)

# Initalize the data generator seperately for the training and validation set
train_generator = get_batches(target_train_data, batch_size, IMAGE_SIZE, word2vec=True)
val_generator = get_batches(target_test_data, batch_size, IMAGE_SIZE, word2vec=True)

# Start Tensorflow session
with tf.Session() as sess:

  # Initialize all variables
  sess.run(tf.global_variables_initializer())

  # Load the pretrained weights into the non-trainable layer
  #saver.restore(sess, 'checkpoints_devise/model_epoch2.ckpt')
  #previous_loader.restore(sess, 'checkpoints_old2/model_epoch42.ckpt')

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
    test_loss = 0.
    test_count = 0
    
    for batch_tx, batch_ty in val_generator:
        new_loss = sess.run(loss, feed_dict={x: batch_tx,
                                                y: batch_ty})
        if math.isnan(new_loss):
              R = build_all_labels_repr()
              proj1 = tf.diag_part(tf.matmul(model_output, tf.transpose(y)))
              sum1 =  (-1)*proj1
              sum2 = tf.matmul(model_output, tf.transpose(R))
              sum3 = tf.transpose(sum1 + tf.transpose(sum2))
              mean = tf.reduce_mean(sum3)
              reg_term = tf.norm(tf.reduce_mean(model_output, 0) - tf.reduce_mean(R, 0))
              variance = tf.pow(tf.reduce_mean(tf.norm(model_output, axis=1)) - tf.reduce_mean(tf.norm(R, axis=1)) ,2)
              reg_term2 = variance
              final_loss = mean + 0.2*reg_term + 0.8*reg_term2
              aa = sess.run([proj1, sum2, sum3, mean, reg_term, variance], feed_dict={x: batch_tx, y: batch_ty})
              print('AA', aa)
        test_loss += new_loss
        test_count += 1
    test_loss /= test_count

    print_in_file("Validation Loss = %s %.4f" % (datetime.now(), test_loss))
    
    # Reset the file pointer of the image data generator
    train_generator = get_batches(target_train_data, batch_size, IMAGE_SIZE, word2vec=True)
    val_generator = get_batches(target_test_data, batch_size, IMAGE_SIZE, word2vec=True)

    print_in_file("{} Saving checkpoint of model...".format(datetime.now()))

    #save checkpoint of the model
    checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch)+'.ckpt')
    save_path = saver.save(sess, checkpoint_name)

    print_in_file("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
