import tensorflow as tf
from layers import *
from tensorflow.contrib.layers import fully_connected, flatten

class Discriminator(object):
    def __init__(self, list_filters, kernel=3, strides=2):
        self.list_filters = list_filters
        self.kernel = kernel
        self.strides = strides

    def network(self, input, reuse=False):
        """

        :param input: tensor
        :return:
        """
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            conv1 = conv1d(input, num_filter=self.list_filters[0], kernel=self.kernel, strides=self.strides, batch_norm=True, activation='leaky_relu')
            print('----------------conv', conv1)
            conv2 = conv1d(conv1, num_filter=self.list_filters[1], kernel=self.kernel, strides=self.strides, batch_norm=True, activation='leaky_relu')
            conv3 = conv1d(conv2, num_filter=self.list_filters[2], kernel=self.kernel, strides=self.strides, batch_norm=True, activation='leaky_relu')
            conv4 = conv1d(conv3, num_filter=self.list_filters[3], kernel=self.kernel, strides=self.strides,batch_norm=True, activation='leaky_relu')
            conv5 = conv1d(conv4, num_filter=self.list_filters[4], kernel=self.kernel, strides=self.strides,batch_norm=True, activation='leaky_relu')
            conv6 = conv1d(conv5, num_filter=self.list_filters[5], kernel=self.kernel, strides=self.strides,batch_norm=True, activation='leaky_relu')
            conv7 = conv1d(conv6, num_filter=self.list_filters[6], kernel=self.kernel, strides=self.strides,batch_norm=True, activation='leaky_relu')
            conv8 = conv1d(conv7, num_filter=self.list_filters[7], kernel=self.kernel, strides=self.strides, batch_norm=True, activation='leaky_relu')
            conv9 = conv1d(conv8, num_filter=self.list_filters[8], kernel=self.kernel, strides=self.strides, batch_norm=True, activation='leaky_relu')
            conv10 = conv1d(conv9, num_filter=self.list_filters[9], kernel=self.kernel, strides=self.strides, batch_norm=True, activation='leaky_relu')
            conv11 = conv1d(conv10, num_filter=self.list_filters[10], kernel=self.kernel, strides=self.strides, batch_norm=True, activation='leaky_relu')
            # conv12 = conv1d(conv11, num_filter=self.list_filters[11], kernel=self.kernel, strides=self.strides, batch_norm=True)
            conv1_1 = conv1d(conv11, num_filter=self.list_filters[10], kernel=1, strides=1, batch_norm=True, activation='leaky_relu')
            ft = flatten(conv1_1)
            fc1 = fully_connected(ft, num_outputs=1, activation_fn=tf.nn.sigmoid)
            print('--------------------------------------done')
        return fc1




