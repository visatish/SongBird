import numpy as np
import tensorflow as tf
from random import shuffle
import pickle
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.mlab import PCA
path_name = '../Sheit/Data/PlaylistSequence.pickle'
input_size = 20000
hidden_layer_size = 500
window_size = 401
num_epochs = 2
drop_out = 0.5
batch_size = 100
learning_rate = 0.01
list_of_weights = []
iterations = 0
fig = plt.figure()
ax = p3.Axes3D(fig)
# Setting the axes properties
#ax.set_xlim3d([-1.0, 1.0])
ax.set_xlabel('X')
#ax.set_ylim3d([-1.0, 1.0])
ax.set_ylabel('Y')
#ax.set_zlim3d([-1.0, 1.0])
ax.set_zlabel('Z')
def pickle_open(path_name):
    f =  open(path_name, 'rb')
    return pickle.load(f)
'''
Simple Two Layer Neural Network that maps input one-hot encoding to output one-hot-encoding
The embedding is the the weight matrix in the last layer
Loss function is softmax-loss
'''
class SkipGramNetwork(object):
    def __init__(self, input_size, hidden_layer_size, window_size, learning_rate = 0.1, drop_out = 0.5):
        self._input_size = input_size
        self._output_size = input_size
        self._window_size = window_size
        self._drop_out = drop_out
        self._hidden_layer_size = hidden_layer_size
        self._learning_rate = learning_rate
        self._sess = None
        self._graph = tf.Graph()
        self._weights = {}
        self._bias = {}
        with tf.name_scope('weights'):
            self._weights['layer1'] = tf.Variable(tf.random_normal([input_size, hidden_layer_size]))
            self._weights['layer2'] = tf.Variable(tf.random_normal([hidden_layer_size, input_size]))
        with tf.name_scope('bias'):
            self._bias['layer1'] = tf.Variable(tf.random_normal([hidden_layer_size]))
            self._bias['layer2'] = tf.Variable(tf.random_normal([input_size]))
        self._saver = tf.train.Saver()
    def open_session(self):
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
    def close_session(self):
        self._sess.close()
    def build_graph(self):
        with tf.name_scope('placeholders'):
            self._x = tf.placeholder(tf.float32, [None, self._input_size], name = "x")
            self._y = tf.placeholder(tf.float32, [None, self._output_size], name = "y")
            self._keep_prob = tf.placeholder(tf.float32, name = "keep_prob")
        with tf.name_scope('layers'):
            self._hidden_layer = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(self._x, self._weights['layer1']), self._bias['layer1'])), self._keep_prob)
            self._output = tf.nn.dropout(tf.add(tf.matmul(self._hidden_layer, self._weights['layer2']), self._bias['layer2']), self._keep_prob)
        with tf.name_scope('loss'):
            self._yhat = tf.nn.softmax_cross_entropy_with_logits(logits=self._output, labels=self._y)
            self._loss = tf.reduce_mean(self._yhat)
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self._loss)
    def train(self, path_name):
        global iterations
        global list_of_weights
        self._data = pickle_open(path_name)
        self._train = self._data[0: int(0.8*len(self._data))] 
        self._test = self._data[int(0.8*len(self._data)):] 
        self._buffer = []
        self._playlist_index =0
        counter =0
        while counter < num_epochs:
            # Obtain batch (Acts like an iterator)
            batch = []
            for p in range(0, batch_size):
                if len(self._buffer) ==0:
                    self._playlist_index+=1
                    if(self._playlist_index >= len(self._train)):
                        shuffle(self._train)
                        self._playlist_index =0
                        counter+=1
                    playlist = self._train[self._playlist_index]
                    temp = int(self._window_size/2)
                    for i in range(0, len(playlist)):
                        for j in range(-temp, temp + 1):
                            if i+j>=0 and j!=0 and i+j<len(playlist):
                                self._buffer.append([playlist[i], playlist[i+j]])
                    shuffle(self._buffer)
                batch.append(self._buffer.pop())    
            # One Hot Encodings
            batch_x = np.zeros((len(batch), self._input_size))
            batch_y = np.zeros((len(batch), self._input_size))
            for i in range(0, len(batch)):
                batch_x[i][batch[i][0]] = 1
                batch_y[i][batch[i][1]] = 1
            self._sess.run(self._optimizer, feed_dict={self._x: batch_x, self._y: batch_y, self._keep_prob: drop_out})
            loss = self._sess.run(self._loss, feed_dict={self._x: batch_x, self._y: batch_y, self._keep_prob: 1})
            print('Iterations',iterations,'Minibatch Loss: {:.6f}'.format(loss))
            if(iterations%20==0):
                #fig.clf()
                ax.set_xlim3d([-40.0, 40.0])
                ax.set_xlabel('Iteration='+str(iterations))
                ax.set_ylim3d([-40.0, 40.0])
                ax.set_ylabel('Y')
                ax.set_zlim3d([-40.0, 40.0])
                ax.set_zlabel('Z')
                
                temp1 = self._weights['layer2'].eval(session = self._sess).T
                myPCA = PCA(temp1)
                feature_matrix = myPCA.Y
                lines = ax.scatter(feature_matrix[:,0], feature_matrix[:,1], feature_matrix[:,2],c="blue")
                figure_title = 'Skip Gram PCA, iteration='+str(iterations)
                
            
                plt.savefig('imgs/plot'+str(iterations)+'.png')
                plt.cla()
            iterations+=1
        
    def test(self):
        self._buffer = []
        self._playlist_index =0
        counter =0
        num_batches =0
        loss =0
        while self._playlist_index < len(self._test):
            # Obtain batch (Acts like an iterator)
            batch = []
            for p in range(0, batch_size):
                if len(self._buffer) ==0:
                    self._playlist_index+=1
                    if(self._playlist_index >= len(self._test)):
                        break
                    playlist = self._test[self._playlist_index]
                    temp = int(self._window_size/2)
                    for i in range(0, len(playlist)):
                        for j in range(-temp, temp + 1):
                            if i+j>=0 and j!=0 and i+j<len(playlist):
                                self._buffer.append([playlist[i], playlist[i+j]])
                    shuffle(self._buffer)
                batch.append(self._buffer.pop())    
            if(len(batch)==0):
                continue
            # One Hot Encodings
            batch_x = np.zeros((len(batch), self._input_size))
            batch_y = np.zeros((len(batch), self._input_size))
            for i in range(0, len(batch)):
                batch_x[i][batch[i][0]] = 1
                batch_y[i][batch[i][1]] = 1
            num_batches +=1
            loss += self._sess.run(self._loss, feed_dict={self._x: batch_x, self._y: batch_y, self._keep_prob: 1})
        print('Test Loss: {:.6f}'.format(loss/num_batches))
    def save(self):
        embedding = self._weights['layer2'].eval(session = self._sess).T
        np.save("EmbeddedMatrix", embedding)
        
def main():
    network = SkipGramNetwork(input_size, hidden_layer_size, window_size, 0.2, drop_out)
    network.build_graph()
    network.open_session()
    network.train(path_name)
    network.test()
    network.save()
    network.close_session()
if __name__ == "__main__":main()