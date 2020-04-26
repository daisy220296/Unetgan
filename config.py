# SEGAN
'''batch_size = 400
canvas_size = 2**14
preemph = 0.95
filename_queue = '../data/segan.tfrecords'
list_filters = [16, 32, 32, 64, 64, 128, 128, 256, 256, 521, 1024]
list_filters = [16, 32, 32, 64, 64, 128, 128, 256, 256, 521, 1024]
learning_rate = 0.0002
momentum = 0.9
iters_d = 1
num_iters = 100
mean = 0
std = 1.0
display = 20
num_epoches = 100
n_input = 16384
n_noise = 16384
lamda = 80
epoch_stop = 40'''
#UNetGAN
batch_size = 150
canvas_size = 2**14
preemph = 0.95
filename_queue = '/home/ubuntu/data/segan.tfrecords'
list_filter = [24, 48, 72, ]
# list_filters = [16, 32, 32, 64, 64, 128, 128, 256, 256, 521, 1024]
learning_rate = 0.0002
momentum = 0.9
iters_d = 1
num_iters = 100
mean = 0
std = 1.0
display = 20
num_epoches = 200
n_input = 16384
n_noise = 16384
lamda = 20
epoch_stop = 200
use_dilated_D = False
use_advance_G = False
basic_channel_d = 24
basic_channel_g = 24
list_rate = [1,2,4]
num_block_d = 3
num_block_g = 8
num_dila_block = 3


