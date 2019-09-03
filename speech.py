import pylab
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
matplotlib.rcParams['figure.figsize'] = [15,10]
import librosa
import librosa.display
import IPython.display as ipd

from PIL import Image
from pydub import AudioSegment
from pydub.utils import mediainfo

data = np.memmap("KsponSpeech_000001.pcm", dtype='h', mode='r')
samp_freq = 16*1024


print(data)

#pylab.plot(data)
#pylab.show()

# Normilze to max amplitude of 1.

speech_samples_norm = data/np.max(data)

strt_samp = 0
end_samp = len(speech_samples_norm)
end_ms = len(speech_samples_norm)/samp_freq

xrange = np.linspace(0, end_ms, end_samp-strt_samp)

# Plot speech and the corresponding spectrogram
fg1 = plt.figure(figsize=(18, 8))
plt.plot(xrange, speech_samples_norm)
plt.xlabel('Time in seconds')
plt.ylabel('Amplitude')
plt.axis('tight')

fg1.savefig('speech1.jpg')

winlen = int(samp_freq*.03)  # Window size of 30 ms
specX = librosa.stft(speech_samples_norm, win_length=winlen)
Xdb = librosa.amplitude_to_db(abs(specX))
fg2 = plt.figure(figsize=(18, 8))
librosa.display.specshow(Xdb, sr=samp_freq, x_axis='time', y_axis='hz', hop_length=winlen/4)

fg2.savefig('specgram.jpg')

print(fg2)

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
drop_out = 0.7
hidden_dim = 1024 #cell의 수
seq_length = 10
word_size = 1 #단어길이

im = Image.open('specgram.jpg')

# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, 1440000])
X_img = tf.reshape(X, [-1, 28, 28, 3])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape=(?, 28, 28, 3)
W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
'''

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
'''

# L3 ImgIn shape=(?, 7, 7, 64)
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
#    Conv      ->(?, 7, 7, 128)
#    Pool      ->(?, 4, 4, 128)
#    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3 = tf.reshape(L3, [-1, 128 * 4 * 4])
'''
Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
'''

# L4 FC 4x4x128 inputs -> 625 outputs
W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
'''
Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
'''

# L5 Final FC 625 inputs -> 10 outputs
W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) + b5
'''
Tensor("add_1:0", shape=(?, 10), dtype=float32)
'''

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_dim)
initial_state = rnn_cell.zero_state()

X_re = tf.reshape(hypothesis, [-1, seq_length, hypothesis.shape[1]])
rnn_output, rnn_state = tf.compat.v1.nn.dynamic_rnn(rnn_cell, X_re, initial_state = initial_state[0], dtype = tf.float32, scope = 'rnn1')

output = tf.contrib.layers.fully_connected(rnn_output, word_size, activation_fn = None)
# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #학습
    for i in range(training_epochs):
        train_output, _cost, _ = sess.run([output, cost, optimizer], feed_dict={X: im, Y: text, keep_prob: drop_out})
        print("step= ",i, "cost = ", _cost)
'''
# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
print('Learning stared. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        c, _, = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

'''