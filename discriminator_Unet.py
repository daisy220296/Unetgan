import tensorflow as tf
from layers import *
from attention import *

class DUnetGan(object):
    def __init__(self, num_block, basic_channel,list_rate, kernelsize = 16, strides=2, use_dilated=False):
        self.num_block = num_block
        self.basic_channel = basic_channel
        self.kernel_size = kernelsize
        self.strides = strides
        self.list_rate = list_rate
        self.use_dilated = use_dilated

    def network(self, input, reuse=False):
        print(input)
        output = input
        channels = self.basic_channel
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            for block in range(self.num_block):
                if self.use_dilated:
                    output = dil_bn_relu_drop(output, variable_scope='dil_conv', kernel_size=self.kernel_size,
                                              channels=channels, rate=self.list_rate[block])
                else:
                    output, _ = conv_bn_lkrelu_down(output, kz=self.kernel_size, strides=self.strides,
                                                 channels=channels, activation='leaky_relu', name='conv_bn_lkrelu_down'+str(block))
                channels = channels * 2

            output = conv1d_cus(output, num_filter=1, kernel=1, strides=1)
            output = tf.nn.sigmoid(output)

        print(output)
        print('----------------')
        return output
