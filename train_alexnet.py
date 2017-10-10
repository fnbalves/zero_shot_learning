#Partialy based on https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

import tensorflow as tf
import numpy as np
import pickle
import os
from models import AlexNet

learning_rate = 0.001
num_epochs = 10
batch_size = 128

dropout_rate = 0.5
num_classes = 50
train_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']

display_step = 1

filewriter_path = '/tmp/train_alexnet/cifar100'
checkpoint_path = '/tmp/finetune_alexnet/'

IMAGE_SIZE = 32

if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

x = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

model = AlexNet(x, keep_prob, num_classes, train_layers)
score = model.fc8

var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                            logits = score, labels = y))

with tf.name_scope('train'):
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

for var in var_list:
    tf.summary.histogram(var.name, var)

tf.summary.scalar('cross_entropy', loss)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scallar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(train_file,
                                     horizontal_flip = True, shuffle = True)
val_generator = ImageDataGenerator(val_file, shuffle = False)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)


# Start Tensorflow session
with tf.Session() as sess:

  # Initialize all variables
  sess.run(tf.global_variables_initializer())

  # Add the model graph to TensorBoard
  writer.add_graph(sess.graph)

  # Load the pretrained weights into the non-trainable layer
  model.load_initial_weights(sess)

  print("{} Start training...".format(datetime.now()))
  print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                    filewriter_path))

  # Loop over number of epochs
  for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        step = 1

        while step < train_batches_per_epoch:

            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size)

            # And run the training op
            sess.run(train_op, feed_dict={x: batch_xs,
                                          y: batch_ys,
                                          keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file
            if step%display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: batch_xs,
                                                        y: batch_ys,
                                                        keep_prob: 1.})
                writer.add_summary(s, epoch*train_batches_per_epoch + step)

            step += 1

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            acc = sess.run(accuracy, feed_dict={x: batch_tx,
                                                y: batch_ty,
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))

        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()

        print("{} Saving checkpoint of model...".format(datetime.now()))

        #save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
