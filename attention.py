import tensorflow as tf
# from layers import *
from tensorflow.compat.v1.layers import conv1d as conv

def self_attention_1d(input, attention_size, name):
    hidden_size = input.shape[2].value
    initializer = tf.random_normal_initializer(stddev=0.1)
    w_omega = tf.get_variable(name= name + 'w_omega', shape=[hidden_size, attention_size], initializer=initializer)
    b_omega = tf.get_variable(name=name+ 'b_omega', shape=[attention_size], initializer=initializer)
    u_omega = tf.get_variable(name=name+'u_omega', shape=[attention_size], initializer=initializer)
    with tf.name_scope('v'):
        v = tf.tanh(tf.tensordot(input, w_omega, axes=1) + b_omega)
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')
    #output = tf.reduce_sum(input * tf.expand_dims(alphas, -1), 1)
    print('vu', vu)
    # print('output', output)
    print('nhan',input * tf.expand_dims(alphas, -1))
    output = input * tf.expand_dims(alphas, -1)
    return output


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

def self_attention_block(x, scope='attention', num_heads=8, sn=False, block='0', name=None):
    """Contains the implementation of Self-Attention block.
    As described in "Self-Attention Generative Adversarial Networks" (SAGAN) https://arxiv.org/pdf/1805.08318.pdf.
    """

    ch = x.shape[-1]
    with tf.variable_scope(scope+name):
        f = conv1d_cus(x, ch // num_heads, kernel=1, strides=1, activation=None, batch_norm=False)  # [bs, h, w, c']
        g = conv1d_cus(x, ch // num_heads, kernel=1, strides=1, activation=None, batch_norm=False)  # [bs, h, w, c']
        h = conv1d_cus(x, ch, kernel=1, strides=1, activation=None, batch_norm=False)  # [bs, h, w, c]
        print('------------------f', f)
        print('------------------g', g)
        print('------------------h', h)
        # N = h * w
        s = g * f
        print('----------s', s)
        assert False

        #s = tf.matmul(g, f, transpose_b=True)  # # [bs, N, N]
        # print('g', g)

        beta = tf.nn.softmax(s, dim=-1)  # attention map

        o = tf.matmul(beta, h)  # [bs, N, C]
        # gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=tf.shape(x))  # [bs, h, w, C]
        x = o + x
        # x = gamma * o + x

    return x