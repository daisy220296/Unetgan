from __future__ import print_function
import tensorflow as tf
# from ops import *
import numpy as np


def pre_emph(x, coeff=0.95):
    x0 = tf.reshape(x[0], [1,])
    diff = x[1:] - coeff * x[:-1]
    concat = tf.concat([x0, diff], 0)
    return concat

def de_emph(y, coeff=0.95):
    if coeff <= 0:
        return y
    x = np.zeros(y.shape[0], dtype=np.float32)
    x[0] = y[0]
    for n in range(1, y.shape[0], 1):
        x[n] = coeff * x[n - 1] + y[n]
    return x

def read_and_decode(filename, canvas_size, preemph=0.):
    filename_queue = tf.train.string_input_producer([filename])
    # raw_dataset = tf.data.TFRecordDataset(filenames)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                'wav_raw': tf.FixedLenFeature([], tf.string),
                'noisy_raw': tf.FixedLenFeature([], tf.string),
            })
    wave = tf.decode_raw(features['wav_raw'], tf.int32)
    _wave = wave
    wave.set_shape(canvas_size)
    wave = (2./65535.) * tf.cast((wave - 32767), tf.float32) + 1.
    noisy = tf.decode_raw(features['noisy_raw'], tf.int32)
    _noisy = noisy
    noisy.set_shape(canvas_size)
    noisy = (2./65535.) * tf.cast((noisy - 32767), tf.float32) + 1.
    print(noisy)
    print(wave)

    if preemph > 0:
        wave = tf.cast(pre_emph(wave, preemph), tf.float32)
        noisy = tf.cast(pre_emph(noisy, preemph), tf.float32)

    return wave, noisy
if __name__ == '__main__':
    import config as cf
    noisy_, wave_, w, n = read_and_decode(cf.filename_queue, cf.canvas_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        noisys = []
        waves = []
        _ws = []
        _ns = []
        for i in range(100):
            noisy, wave, _w, _n = sess.run([noisy_, wave_, w, n])
            noisys.append(noisy)
            waves.append(wave)
            _ws.append(_w)
            _ns.append(_n)
        # print(min(noisys), max(noisys))
        # print(min(waves), max(waves))
        # print(min(_ws), max(_ws))
        # print(min(_ns), max(_ns))
        import pdb; pdb.set_trace()
        # assert False
            # import pdb; pdb.set_trace()
        coord.request_stop()
        coord.join(threads)
    assert False
