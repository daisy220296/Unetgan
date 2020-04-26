from train import *
import tensorflow as tf
from scipy.io import wavfile
import numpy as np
import config as cf
from pystoi.stoi import stoi
from pypesq import pesq
import os

def test(data, segan):
    x = (2./65535.) * (data.astype(np.float32) - 32767) + 1.
    input_ = np.zeros((cf.batch_size, cf.n_input, 1), dtype=float)
    n_size = int(len(x) / cf.n_input)
    #import pdb; pdb.set_trace()
    for i in range(n_size):
        start = i * cf.n_input
        end = (i+1)* (cf.n_input )
        input_[i] = np.reshape(x[start : end], newshape=(cf.n_input, 1))
    if n_size * cf.n_input > len(x):
        pad = len(input_) - n_size * cf.n_input
        input_[n_size][:pad] = input_[n_size * cf.n_input:len(x)]
    output_g = segan.sess.run(segan.g_output, feed_dict={segan.Z: input_})
    output = np.zeros_like(x, dtype=float)
    for i in range(n_size):
        start = i * cf.n_input
        end = (i+1) * (cf.n_input)
        output[start: end] = np.reshape(output_g[i], newshape=(cf.n_input,))
    if n_size * cf.n_input > len(x):
        output[cf.n_input * n_size : len(x)] = np.reshape(output[n_size][:pad], newshape=(cf.pad,))
    output = (output -1)* 65535. /2. + 32767
    return x, output
if __name__=='__main__':
    folder_clean = '/home/ubuntu/data/clean_test_16'
    folder_noise = '/home/ubuntu/data/noisy_test_16'
    list_file = os.listdir(folder_clean)
    
    #data = wavfile.read('/home/ubuntu/data/noise/p287_424.wav')
    #clean = wavfile.read('/home/ubuntu/data/clean/p287_424.wav')
    #print(clean[0], data[0])
    #assert False
    '''data = data[1]
    clean = clean[1]
    clean =  (2./65535.) * (clean.astype(np.float32) - 32767) + 1.'''
    stoi_scores = []
    pesq_scores = []
    fs = 16000
    with tf.Session() as sess:
        segan = SEGAN(sess, type_model='UNetGan')
        segan.build()
        segan.load('save_model')
        for f in list_file:
            file_name_clean = os.path.join(folder_clean, f)
            file_name_noise = os.path.join(folder_noise, f)
            print(file_name_clean)
            print(file_name_noise)
            if not os.path.isfile(file_name_noise):print('break'); continue
            data = wavfile.read(file_name_noise)
            clean = wavfile.read(file_name_clean)
            print(data[0], clean[0])
            data = data[1]
            clean = clean[1]
            #assert False
            #clean = (2./65535.) * (clean.astype(np.float32) - 32767) + 1.
            x, denoised = test(data, segan)
            
            stoi_scores.append(stoi(clean, denoised, fs, extended=True))
            pesq_scores.append(pesq(clean, denoised, fs))
            print(stoi_scores[-1])
            print(pesq_scores[-1])
            print(f)
            print('----------------')
        print('mean stoi',np.mean(stoi_scores))
        print('mean pesq',np.mean(pesq_scores))



