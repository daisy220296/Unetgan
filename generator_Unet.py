import tensorflow as tf
from layers import *
from attention import *
class UNet_GAN(object):
    def __init__(self,input, num_block, dil_block, rate_list, basic_channel, advance_gan=False):
        self.num_block = num_block
        self.dil_block = dil_block
        self.rates = rate_list
        self.basic_channel = basic_channel
        if advance_gan:
            self.network = self.advance_network(input)
        else:
            self.network = self.simple_network(input)


    def simple_network(self, input):
        # downsample network
        with tf.variable_scope('generator'):
            prev_block = []
            output = input
            for block in range(self.num_block):
                output, x_leky = conv_bn_lkrelu_down(output, channels=self.basic_channel * (block + 1),
                                             name='conv_bn_lkrelu_' + str(block+1))
                #x_leky = self_attention_1d(x_leky, self.basic_channel * (block + 1)//4, name='down_sample'+str(block))
                # print('down', output)
                # print('x_leky', x_leky)

                prev_block.append(x_leky)
            #Dilated block
            for block in range(self.dil_block):
                output = dil_1d(output, rate=self.rates[block], name='simple'+str(block))

            # upsample network
            for block in range(self.num_block -1 , -1, -1):
                print(block)
                # concated = tf.concat([output, prev_block[block]])
                output = conv_bn_lkrelu_upsamp(output, prev_block[block], channels=self.basic_channel * (block + 1))

            output = conv1d_cus(output, num_filter=1, kernel=1, strides=1, activation='tanh',
                            batch_norm=False)
        return output, None

    def advance_network(self, input):

        with tf.variable_scope('generator'):
            prev_block = []
            output = input
            for block in range(self.num_block):
                if block == self.num_block -1:
                    end_block = True
                else:
                    end_block = False
                print('block----------------------', block)
                output, output_attent = downsample_block(output, channels=self.basic_channel * (block + 1), end_block=end_block, name='generator'+str(block))
                prev_block.append(output_attent)

            for block in range(self.dil_block):
                output = dil_1d(output, rate=self.rates[block], name=str(block))

            for block in range(self.num_block-1, -1, -1):
                # print('block', block)
                output = upsampling_block(output,prev_block[block], channels=self.basic_channel * (block + 1))

            output = conv1d_cus(output, num_filter=1, kernel=1, strides=1, activation='tanh', batch_norm=False)

        return output, None










