from train import *
import tensorflow as tf
from scipy.io import wavfile
import numpy as np
import config as cf
from pystoi.stoi import stoi
from pypesq import pesq
import os
def pre_emph(x, coeff=0.95):
    x0 = np.reshape(x[0], [1,1])
    diff = x[1:] - coeff * x[:-1]
    #print('x0',x0.shape)
    #print('diff',diff.shape)
    concat = np.concatenate((x0, diff), 0)
    return concat
def de_emph(y, coeff=0.95):
    if coeff <= 0:
        return y
    x = np.zeros(y.shape[0], dtype=np.float32)
    x[0] = y[0]
    for n in range(1, y.shape[0], 1):
        x[n] = coeff * x[n - 1] + y[n]
    return x

def test(data, segan):
    x = (2./65535.) * (data.astype(np.float32) - 32767) + 1.
    input_ = np.zeros((cf.batch_size, cf.n_input, 1), dtype=float)
    n_size = int(len(x) / cf.n_input)
    #import pdb; pdb.set_trace()
    #input_ = pre_emph(input_)
    for i in range(n_size):
        start = i * cf.n_input
        end = (i+1)* (cf.n_input )
        input_[i] = np.reshape(x[start : end], newshape=(cf.n_input, 1))
    if n_size * cf.n_input > len(x):
        pad = len(input_) - n_size * cf.n_input
        input_[n_size][:pad] = input_[n_size * cf.n_input:len(x)]
    for i in range(input_.shape[0]):
        #print(input_[i])
        input_[i] = pre_emph(input_[i])
    output_g = segan.sess.run(segan.g_output, feed_dict={segan.Z: input_})
    output_g = (output_g -1)* 65535. /2. + 32767
    output_g = np.reshape(output_g, (150,16384, 1))
    for i in range(n_size+1):
        #print(output_g[i].shape,de_emph(output_g[i]).shape)
        output_g[i]= np.reshape(de_emph(output_g[i]), [16384, 1])
    #output_g = np.reshape(de_emph(output_g),[400,16384,1])
    output = np.zeros_like(x, dtype=float)
    for i in range(n_size):
        start = i * cf.n_input
        end = (i+1) * (cf.n_input)
        output[start: end] = np.reshape(output_g[i], newshape=(cf.n_input,))
    if n_size * cf.n_input > len(x):
        output[cf.n_input * n_size : len(x)] = np.reshape(output[n_size][:pad], newshape=(cf.pad,))
    #output = (output -1)* 65535. /2. + 32767
    #output = de_emph(output)

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
        print('-------------------------------done build')
        segan.load('model_unet_att')
        print('-------------------------------done load')
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



