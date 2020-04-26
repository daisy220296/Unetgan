from train import *
import tensorflow as tf

if __name__ == '__main__':
    with tf.Session() as sess:
        gan = SEGAN(sess, type_model='UNetGan')
        gan.train(save_model='model_unet_premh')
