import tensorflow

#Based on https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
class AlexNet(object):
    def __init__(self, x, keep_prob, num_classes, skip_layer):
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.IS_TRAINING = False
        

    def create(self):
        pass

    def load_initial_weights(self):
        pass

    #Convolution function that can be split in multiple GPUS
    def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
             padding='SAME', groups=1):

        #Get number of input chennels
        input_channels = int(x.get_shape()[-1])

        convolve = lambda i, k: tf.conv2d(i, k, strides = [1, stride_y, stride_x, 1],
                                          padding = padding)

        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights',
					shape = [filter_height, filter_width, 
						input_channels/groups, num_filters])
	        biases = tf.get_variable('biases', shape = [num_filters])

            if gourps == 1:
                conv = convolce(x, weights)
            else:
                input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
                weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

                conv = tf.concat(axis=3, values=output_groups)

            bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().ad_list())
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
    
