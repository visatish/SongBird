import tensorflow as tf
import numpy as np
import pickle

path_name = '../Sheit/Data/PlaylistSequence.pickle'
lstm_size = 500
num_layers = 3
num_steps = 50
batch_size = 10
num_features = 20001
keep_prob = 0.5
max_time_length = 249

def pickle_open(path_name):
	f =  open(path_name, 'rb')
	return pickle.load(f)

def LSTM_cell(keep_prob):
	return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=0.0), output_keep_prob=keep_prob)

def multi_LSTM_cell(num_layers, keep_prob):
	return tf.contrib.rnn.MultiRNNCell([LSTM_cell(keep_prob) for i in range(num_layers)])


# Batchsize x Timestep x Number of Features (20k features due to one hot encoding)
x_input = tf.placeholder(tf.float32, [None, max_time_length, num_features], name='input')
y_input = tf.placeholder(tf.float32, [None, 249, num_features], name='labels')


cell = multi_LSTM_cell(num_layers, keep_prob)
output, state = tf.nn.dynamic_rnn(cell, x_input, dtype=tf.float32)

output = tf.transpose(output, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0]) - 1)

out_size = y_input.get_shape()[2].value
logit = tf.contrib.layers.fully_connected(
    output, out_size, activation_fn=None)
prediction = tf.nn.softmax(logit)

flat_target = tf.reshape(y_input, [-1] + y_input.shape.as_list()[2:])
flat_logit = tf.reshape(logit, [-1] + logit.shape.as_list()[2:])
loss = tf.losses.softmax_cross_entropy(flat_target, flat_logit)
loss = tf.reduce_mean(loss)


optimize = tf.train.AdamOptimizer(learning_rate=0.03).minimize(loss)

data = pickle_open(path_name)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	counter =0
	while counter< len(data):
		batch_x = np.zeros([batch_size, max_time_length, num_features])
		batch_y = np.zeros([batch_size, max_time_length, num_features])
		for j in range(0, batch_size):
			x = data[counter]
			y = data[counter][1:]
			#x = data[counter][0:max_time_length]
			#y = data[counter][1:max_time_length]
			for i in range(0, max_time_length-len(x)):
				x.append(20001)
				y.append(20001)
			y.append(20001)
			for i in range(0, max_time_length):
				#batch_x[j][i][i%3] = 1
				#batch_y[j][i][i%3] = 1
				batch_x[j][i][x[i]-1] = 1
				batch_y[j][i][y[i]-1] = 1
			counter+=1
		l, opt, out = sess.run([loss, optimize, output], feed_dict={x_input: batch_x, y_input: batch_y})
		print(out.shape)
		exit(0)
		print("Iteration {} , LSTM Loss {:.6f}".format(int(counter/batch_size), l))





