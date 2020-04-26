import tensorflow as tf
import numpy as np
from layers import *
import config as cf

class Unet(object):
    def __init__(self, list_filters, kernel=31, strides=2,  num_block=12, batch_size=cf.batch_size):
        self.list_filters = list_filters
        self.kernel = kernel
        self.num_block = num_block
        self.strides = strides
        self.batch_size = batch_size
        self.z = tf.random_normal(shape=(batch_size, 8, 1024), mean=cf.mean, stddev=cf.std,
                                     name='z', dtype=tf.float32)

    def network_simple(self, input):
        """

        :param input:
        :return:
        """
        # assert (len(self.list_filters) != num_block)
        #  encoder
        with tf.variable_scope('generator') as scope:
            z = self.z

            with tf.variable_scope('sampling'):
                conv1 = conv1d(input, num_filter=self.list_filters[0], kernel=self.kernel, strides=self.strides)


                conv2 = conv1d(conv1, num_filter=self.list_filters[1], kernel=self.kernel, strides=self.strides)
                conv3 = conv1d(conv2, num_filter=self.list_filters[2], kernel=self.kernel, strides=self.strides)
                conv4 = conv1d(conv3, num_filter=self.list_filters[3], kernel=self.kernel, strides=self.strides)
                conv5 = conv1d(conv4, num_filter=self.list_filters[4], kernel=self.kernel, strides=self.strides)
                conv6 = conv1d(conv5, num_filter=self.list_filters[5], kernel=self.kernel, strides=self.strides)
                conv7 = conv1d(conv6, num_filter=self.list_filters[6], kernel=self.kernel, strides=self.strides)
                conv8 = conv1d(conv7, num_filter=self.list_filters[7], kernel=self.kernel, strides=self.strides)

                conv9 = conv1d(conv8, num_filter=self.list_filters[8], kernel=self.kernel, strides=self.strides)
                conv10 = conv1d(conv9, num_filter=self.list_filters[9], kernel=self.kernel, strides=self.strides)
                conv11 = conv1d(conv10, num_filter=self.list_filters[10], kernel=self.kernel, strides=self.strides)

            # Decoder
            with tf.variable_scope('upsampling'):
                concat11 = tf.concat([conv11, z], axis=2)
                deconv10 = deconv(concat11, num_filters=self.list_filters[10], kernel=self.kernel, strides=self.strides)
                deconv10 = tf.layers.dropout(deconv10)
                concat10 = tf.concat([deconv10, conv10], axis=2)

                deconv9 = deconv(concat10, num_filters=self.list_filters[9], kernel=self.kernel, strides=self.strides)
                deconv9 = tf.layers.dropout(deconv9)
                concat9 = tf.concat([deconv9, conv9], axis=2)

                deconv8 = deconv(concat9, num_filters=self.list_filters[8], kernel=self.kernel, strides=self.strides)
                deconv8 = tf.layers.dropout(deconv8)
                concat8 =  tf.concat([deconv8, conv8], axis=2)

                deconv7 = deconv(concat8, num_filters=self.list_filters[7], kernel=self.kernel, strides=self.strides)
                deconv7 = tf.layers.dropout(deconv7)
                concat7 = tf.concat([deconv7, conv7], axis=2)

                deconv6 = deconv(concat7, num_filters=self.list_filters[6], kernel=self.kernel, strides=self.strides)
                deconv6 = tf.layers.dropout(deconv6)
                concat6 = tf.concat([deconv6, conv6], axis=2)

                deconv5 = deconv(concat6, num_filters=self.list_filters[5], kernel=self.kernel, strides=self.strides)
                deconv5 = tf.layers.dropout(deconv5)
                concat5 = tf.concat([deconv5, conv5], axis=2)

                deconv4 = deconv(concat5, num_filters=self.list_filters[4], kernel=self.kernel, strides=self.strides)
                deconv4 = tf.layers.dropout(deconv4)
                concat4 = tf.concat([deconv4, conv4], axis=2)

                deconv3 = deconv(concat4, num_filters=self.list_filters[3], kernel=self.kernel, strides=self.strides)
                deconv3 = tf.layers.dropout(deconv3)
                concat3 = tf.concat([deconv3, conv3], axis=2)

                deconv2 = deconv(concat3, num_filters=self.list_filters[2], kernel=self.kernel, strides=self.strides)
                deconv2 = tf.layers.dropout(deconv2)
                concat2 = tf.concat([deconv2, conv2], axis=2)

                deconv1 = deconv(concat2, num_filters=self.list_filters[1], kernel=self.kernel, strides=self.strides)
                deconv1 = tf.layers.dropout(deconv1)
                concat1 = tf.concat([deconv1, conv1], axis=2)

                deconv1 = deconv(concat1, num_filters=self.list_filters[0], kernel=self.kernel, strides=self.strides)
                output = conv1d(deconv1, num_filter=1, kernel=1, strides=1, activation=tf.nn.tanh)
                print(output)
                # assert False

        return output, z

