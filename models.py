#Based on https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

import tensorflow as tf
import numpy as np


#Convolution function that can be split in multiple GPUS
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
             padding='SAME', groups=1, verbose_shapes=False):

    #Get number of input chennels
    input_channels = int(x.get_shape()[-1])

    if verbose_shapes:
        print('INPUT_CHANNELS', input_channels)
        print('X SHAPE conv', x.get_shape())

    convolve = lambda i, k: tf.nn.conv2d(i, k, strides = [1, stride_y, stride_x, 1],
                                          padding = padding)

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels/groups, num_filters], trainable=True)
        biases = tf.get_variable('biases',shape=[num_filters], trainable=True)

        if groups == 1:
            conv = convolve(x, weights)
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            conv = tf.concat(axis=3, values=output_groups)

        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        relu = tf.nn.relu(bias, name = scope.name)

        return relu

#Full connected layer
def fc(x, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu == True:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act
    
#Max pooling layer
def max_pool(x, filter_height, filter_width, stride_y, stride_x,
                name, padding='SAME', verbose_shapes=False):
    if verbose_shapes:
        print('X SHAPE maxpool', x.get_shape())

    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                              strides = [1,stride_y, stride_x, 1],
                              padding = padding, name = name)

#Batch normalization
def lrn(x, radius, alpha, beta, name, bias=1.0, verbose_shapes=False):
    if verbose_shapes:
        print('X SHAPE lrn', x.get_shape())

    return tf.nn.local_response_normalization(x, depth_radius = radius,
                                                  alpha = alpha, beta = beta,
                                                  bias = bias, name = name)

#Dropout layer
def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

class AlexNet(object):
    def __init__(self, x, keep_prob, num_classes, skip_layer, pre_trained_path=None):
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.IS_TRAINING = False
        self.WEIGHTS_PATH = pre_trained_path
        self.create()

    def create(self):
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = conv(self.X, 11, 11, 96, 4, 4, padding = 'VALID', name = 'conv1')
        norm1 = lrn(conv1, 2, 2e-05, 0.75, name = 'norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')

        # 2nd Layer: Conv (w ReLu) -> Lrn -> Poolwith 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups = 2, name = 'conv2')
        norm2 = lrn(conv2, 2, 2e-05, 0.75, name = 'norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding = 'VALID', name ='pool2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name = 'conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups = 2, name = 'conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups = 2, name = 'conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name = 'fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations
        # (for tf.nn.softmax_cross_entropy_with_logits)
        self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu = False, name='fc8')
        

    def load_pre_trained_weights(self, session):
        weights_dict = np.load(self.WEIGHTS_PATH, encoding = 'bytes').item()

        for op_name in weights_dict:
            if op_name not in self.SKIP_LAYER:
                with tf.variable_scope(op_name, reuse=True):
                    for data in weights_dict[op_name]:
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))

        
