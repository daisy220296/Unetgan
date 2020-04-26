import tensorflow as tf
from tensorflow.compat.v1.layers import conv1d as conv
from tensorflow.contrib.layers import fully_connected
import numpy as np
from attention import *
def conv1d_cus(input, num_filter, kernel=3, strides=1, activation='prelu', batch_norm=True):

    x = conv(input, num_filter, kernel, strides=strides, padding='same',
             kernel_initializer=tf.keras.initializers.he_normal())
    if activation == 'prelu':
        x = prelu(x)
    if activation == 'leaky_relu':
        x = tf.nn.leaky_relu(x, alpha=0.3)
    elif activation == 'relu':
        x = tf.nn.relu(x)
    elif activation == 'identity':
        x = tf.identity(x)
    elif activation == 'tanh':
        x = tf.nn.tanh(x)
    if batch_norm:
        x = tf.compat.v1.layers.batch_normalization(x)
    return x
def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)
def conv2d(input, num_filter, kernel, activation='relu'):
    if activation == 'leaky_relu':
        activation = tf.nn.leaky_relu
    else:
        activation = tf.nn.relu
    x = tf.layers.conv2d(input, num_filter, kernel, activation=activation,
                         kernel_initializer=tf.keras.initializers.he_normal())
    return x
def deconv(x, num_filters, kernel=31,strides=2, activation='keaky_relu' ):
    if activation == 'leaky_relu':
        activation = tf.nn.leaky_relu
    else:
        activation = tf.nn.relu
    shape_x = x.get_shape()
    # print('dau--------------------',shape_x)
    x = tf.reshape(x, shape=(shape_x[0], shape_x[1], 1, shape_x[2]))
    x = tf.compat.v1.layers.conv2d_transpose(x, filters=num_filters, kernel_size=(kernel, 1), strides=(strides, 1),
                                             activation=activation, padding='same',
                                   kernel_initializer=tf.keras.initializers.he_normal())
    shape_x = x.get_shape()
    # print('sau----------------',shape_x)
    # assert False
    x = tf.reshape(x, [shape_x[0], shape_x[1], shape_x[3]])
    return x


def conv_bn_lkrelu_down(input, kz=15, strides=1, channels=24, activation=None, name=None):
    with tf.variable_scope(name, default_name   ='conv_bn_lkrelu'):
        x = conv1d_cus(input,num_filter=channels, kernel=kz, strides=strides, batch_norm=False, activation=activation )
        x = tf.compat.v1.layers.batch_normalization(x)
        x_leaky = tf.nn.leaky_relu(x, alpha=0.1)
        # print('trc max', x_leaky)
        x = tf.nn.max_pool1d(x_leaky, [2,1,1], [2,1,1], padding='SAME')
    return x, x_leaky
def conv_bn_lkrelu_upsamp(input, pre_layer, kz=5, strides=2, channels=24, name=None):
    with tf.variable_scope(name, default_name='conv_bn_lkrelu_upsamp'):
        # print('input', input)
        # print('prelayer', pre_layer)
        x = deconv(input, num_filters=channels, kernel=2,strides=strides, activation=None )
        # print('x_up', x)
        # print('---------------')
        x = tf.concat([x, pre_layer], axis=2)
        x = conv1d_cus(x, num_filter=channels, kernel=kz, strides=1, batch_norm=False, activation=False)
        x = tf.compat.v1.layers.batch_normalization(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)
    return x


def atrous_conv1d(value, filters, rate, padding, name=None):
    '''wrapper for tf.nn.convolution to atrous_conv1d
    lets you choose the filters
    usage:
        b = tf.placeholder(shape=(None, 1, 4096,1), dtype=tf.float32)
        atrous_conv1d(b, filters=tf.ones((1, 4096, 1)),rate=2, padding='VALID')

        b = tf.placeholder(shape=(None, 4096), dtype=tf.float32)
        atrous_conv1d(b, filters=tf.ones((1, 4096, 1)),rate=2, padding='VALID')

    correctness not guaranteed'''
    # print(rate)
    # print(tf.expand_dims(value, 1))
    # print(tf.expand_dims(filters, 1))
    x = tf.squeeze(tf.nn.convolution(
        input=tf.expand_dims(value, 1),
        filter=tf.expand_dims(filters, 1),
        padding=padding,
        dilation_rate=np.broadcast_to(rate, (2,)),
        name=name), 1)

    # print(x)
    # print('-------------------')
    return x
# b = tf.placeholder(shape=(None, 4096,1), dtype=tf.float32)
# atrous_conv1d(b, filters=tf.ones((1, 1, 2)),rate=2, padding='VALID')
# b = tf.placeholder(shape=(None, 4096), dtype=tf.float32)
# atrous_conv1d(b, filters=tf.ones((1, 4096, 1)),rate=2, padding='VALID')
def dil_1d(input, kz=3, channels=192, rate=1, name=None):
    stddev = 5e-2
    # kernel_shape = (input.get_shape()[0], kz, channels)
    kernel_shape = (kz, input.get_shape()[2], channels)
    kernel = tf.get_variable(name + 'weight', kernel_shape, initializer=tf.random_normal_initializer(stddev=stddev))
    x = atrous_conv1d(input, kernel, rate=rate, padding="SAME")
    x = tf.layers.batch_normalization(x)
    x = tf.nn.leaky_relu(x, alpha=0.3)
    return x


def dil_bn_relu_drop( input, variable_scope, kernel_size, channels,rate,
                     use_bn=True, use_dropout=True, keep_prob=1.0):
    with tf.variable_scope(variable_scope):
        x = dil_1d(input, kz=kernel_size, channels=channels, rate=rate, name=variable_scope)
        if use_bn:
            x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        if use_dropout:
            x = tf.layers.dropout(x, keep_prob)
    return x

# def conv_bn_lkrelu(input, kz=15, strides=1, channels=24, activation=None, name=None):
#     with tf.variable_scope(name, default_name   ='conv_bn_lkrelu'):
#         x = conv1d(input,num_filter=channels, kernel=kz, strides=strides, batch_norm=False, activation=activation )
#         x = tf.compat.v1.layers.batch_normalization(x)
#         x_leaky = tf.nn.leaky_relu(x, alpha=0.1)


def downsample_block(input, kernel_size=15, strides=1, channels=24,
                     activation='leaky_relu',rate=1, use_resnet=True, resnet_deep=3, use_attention=True, \
                     end_block=False, ksize=2, stridepool=2, name=None):
    x = dil_bn_relu_drop(input, name,  kernel_size, channels, rate)
    orign = x
    if use_resnet:
        for ares in range(resnet_deep):
            if ares < resnet_deep -1:
                _, x = conv_bn_lkrelu_down(x, kz=kernel_size, strides=strides, channels=channels, activation=activation)
            else:
                _, x = conv_bn_lkrelu_down(x, kz=kernel_size, strides=strides, channels=channels, activation='identity')
    x += orign
    if use_attention:
        output_attent = self_attention_1d(x, name=name)
    else:
        output_attent = x
    # if not end_block:
    x = tf.nn.max_pool1d(x, [2,1,1], [2,1,1], padding='SAME', name='pool')
    return x, output_attent

def upsampling_block(input, pre_layer, kernel_size=15, strides=1, channels=24,
                     activation='leaky_relu', use_resnet=True, resnet_deep=3, ksize_deconv=2, stride=2, name=None):
    # print('trc', input)
    #x = conv_bn_lkrelu_upsamp(input, pre_layer, kz=kernel_size, strides=strides, channels=channels)
    x = deconv(input, num_filters=channels, kernel=kernel_size,strides=2)
    # print('sau', x)
    # print('pre', pre_layer)
    # print('------------------------')
    x = tf.concat([x, pre_layer], axis=2)
    # print('x concat', x)
    _, x = conv_bn_lkrelu_down(x, kz=kernel_size, strides=strides, channels=channels, activation=activation)
    orign = x
    if use_resnet:
        for ares in range(resnet_deep):
            if ares < resnet_deep - 1:
                _, x = conv_bn_lkrelu_down(x, kz=kernel_size, strides=strides, channels=channels, activation=activation)
            else:
                _, x = conv_bn_lkrelu_down(x, kz=kernel_size, strides=strides, channels=channels, activation='identity')
    x += orign
    # print('x_cuoi', x)
    # print('-------------------------')
    return x





