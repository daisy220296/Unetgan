import tensorflow as tf
from discriminator import *
from generator import *
import config as cf
from read_tfrecord import *
import os
from scipy.io import wavfile
class SEGAN():

    def __init__(self, sess, type_model='original', retrain=False, model_path='save_model'):
        self.list_filters = cf.list_filters
        self.name = type_model
        if type_model == 'original':
            self.generator = Unet(self.list_filters)

        self.discriminator = Discriminator(self.list_filters)
        self.batch_size = cf.batch_size
        self.canvas_size = cf.canvas_size
        self.preemph = cf.preemph
        self.lr = cf.learning_rate
        self.mmt = cf.momentum
        self.num_iters = cf.num_iters
        self.iters_d = cf.iters_d
        self.sess = sess
        self.num_epoches = cf.num_epoches
        self.retrain = retrain

    def build(self):
        wav, noisy = read_and_decode(cf.filename_queue, cf.canvas_size,
                                     cf.preemph)
        wavbatch, noisybatch = tf.train.shuffle_batch([wav,
                                                       noisy],
                                                      batch_size=cf.batch_size,
                                                      num_threads=2,
                                                      capacity=1000 + 3 * cf.batch_size,
                                                      min_after_dequeue=1000,
                                                      name='wav_and_noisy')
        wavbatch = tf.expand_dims(wavbatch, -1)
        noisybatch = tf.expand_dims(noisybatch, -1)
        self.wavbatch, self.noisebatch = wavbatch, noisybatch
        # print(self.wavbatch, self.noisebatch)
        # assert False

        self.X = tf.placeholder(shape=(cf.batch_size, cf.n_input, 1), dtype=tf.float32)
        self.Z = tf.placeholder(shape=(cf.batch_size, cf.n_noise, 1), dtype=tf.float32)
        self.lamda = tf.placeholder(dtype=tf.float32)
        input_d_real = tf.concat([self.X, self.Z], axis=2)
        g_output, z = self.generator.network_simple(self.Z)
        d_real_out = self.discriminator.network(input_d_real, reuse=False)
        input_d_fake = tf.concat([g_output, self.Z], axis=2)
        d_fake_out = self.discriminator.network(input_d_fake, reuse=True)
        self.loss_real_d = tf.reduce_mean(tf.squared_difference(d_real_out, 1.0))
        self.loss_fake_d = tf.reduce_mean(tf.squared_difference(d_fake_out, 0.0))
        self.loss_d = self.loss_fake_d + self.loss_real_d
        self.g_output = g_output

        self.loss_g_l1 = tf.reduce_mean(tf.abs(g_output - self.X))
        self.loss_g = tf.reduce_mean(tf.squared_difference(d_fake_out, 1.0))
        self.loss_total_g = self.lamda * self.loss_g_l1 + self.loss_g

        self.summary_loss_d = tf.summary.scalar('loss_d', self.loss_d)
        self.summary_loss_g = tf.summary.scalar('loss_g', self.loss_total_g)
        self.audio = tf.summary.audio('output G', g_output, 16e3)
        self.audio_h = tf.summary.histogram('outputG_h', g_output)
        self.real_noise = tf.summary.audio('noise', self.Z, 16e3)
        self.real_noise_h = tf.summary.histogram('noise_h', self.Z)
        self.real_clean = tf.summary.audio('clean', self.X, 16e3)
        self.real_clean_h = tf.summary.histogram('clean_h', self.X)
        self.writer = tf.summary.FileWriter('./log_visual')

        D_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        G_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

        # self.opt_d = tf.train.AdamOptimizer(self.lr, self.mmt).minimize(self.loss_d, var_list=D_var_list)
        # self.opt_g = tf.train.AdamOptimizer(self.lr, self.mmt).minimize(self.loss_total_g, var_list=G_var_list)
        self.opt_d = tf.train.RMSPropOptimizer(self.lr, self.mmt).minimize(self.loss_d, var_list=D_var_list)
        self.opt_g = tf.train.RMSPropOptimizer(self.lr, self.mmt).minimize(self.loss_total_g, var_list=G_var_list)
    def save_model(self, path_save, step):
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()

        self.saver.save(self.sess, os.path.join(path_save, self.name), global_step=step)
    def load(self, saved_path):
        '''if saved_path is not os.path.exists(saved_path):
            print('Model not exit')
            assert False'''

        ckpt = tf.train.get_checkpoint_state(saved_path)
        if ckpt and os.path.basename(ckpt.model_checkpoint_path):
            name_ckpt = os.path.basename(ckpt.model_checkpoint_path)
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(saved_path, name_ckpt))

    def train(self, save_model, load_model=None):
        lamda = cf.lamda
        self.build()
        sum_g = tf.summary.merge([self.summary_loss_g, self.real_clean, self.real_noise, self.audio,
                                  self.real_clean_h, self.real_noise_h, self.audio_h])
        sum_d = tf.summary.merge([self.summary_loss_d])
        init = tf.global_variables_initializer()
        if load_model is None:
            self.sess.run(init)
        else:
            self.load(load_model)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        total_d = []
        total_g = []
        total_loss_l1 = []
        total_loss_only_g = []
        total_loss_fake = []
        total_loss_real = []
        save_path = 'save_file_verse'

        print('Training ..............')
        for epoch in range(self.num_epoches):
            for iter in range(self.num_iters):
                for i in range(cf.iters_d):
                    wav, noise = self.sess.run([self.wavbatch, self.noisebatch])
                    _, loss_d_, loss_real, loss_fake = self.sess.run([self.opt_d, self.loss_d, self.loss_real_d, self.loss_fake_d], feed_dict={self.X:wav, self.Z: noise})
                __, loss_g_, loss_only_g, loss_l1, g = self.sess.run([self.opt_g, self.loss_total_g, self.loss_g, self.loss_g_l1, self.g_output], feed_dict={self.X: wav, self.Z:noise, self.lamda:lamda})
                total_d.append(loss_d_)
                total_g.append(loss_g_)
                total_loss_only_g.append(loss_only_g)
                total_loss_l1.append(loss_l1)
                total_loss_fake.append(loss_fake)
                total_loss_real.append(loss_real)
                for l in range(cf.batch_size):
                    print(g[l])
                    wavfile.write(os.path.join(save_path,'{}_sample_{}.wav'.format(l,l)),48000,g[l])# * 65535 / 2 + 32767)# np.reshape(g[l], (g[l].shape[0],)))
                    wavfile.write(os.path.join(save_path,'{}_noise_{}.wav'.format(l,l)),48000, noise[l])# * 65535 /2 + 32767)
                    wavfile.write(os.path.join(save_path,'{}_clean_{}.wav'.format(l,l)),48000, wav[l])# * 65535 /2 + 32767)
                assert False
                if iter % 50 ==0:
                    _sum_g, _sum_d = self.sess.run([sum_g, sum_d], feed_dict={self.X: wav, self.Z:noise, self.lamda: lamda})
                    self.writer.add_summary(_sum_d, epoch * self.num_iters + iter)
                    self.writer.add_summary(_sum_g, epoch * self.num_iters + iter)
                    assert False
            # if (epoch % 2 == 0):
            #     lamda = 100 * lamda
            self.save_model(path_save=save_model, step=epoch)
            print('Epoch', epoch, '      loss_d:', np.mean(total_d), '         loss_g:', np.mean(total_g),
                  '  loss real ', np.mean(total_loss_real), 'loss fake', np.mean(total_loss_fake), 'loss l1', np.mean(total_loss_l1),
                  '  loss only g', np.mean(total_loss_only_g))
            total_d = []
            total_g = []
            total_loss_fake = []
            total_loss_l1 = []
            total_loss_real = []
            total_loss_only_g = []
            if epoch == cf.epoch_stop:
                lamda = 0.0


        print('Done')
        coord.request_stop()
        coord.join(threads)




if __name__=='__main__':
    with tf.Session() as sess:
        gan = SEGAN(sess)
        gan.train(save_model='save_model', load_model='save_model_0002')
