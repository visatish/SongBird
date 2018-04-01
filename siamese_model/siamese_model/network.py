"""
Siamese Network implementation in Tensorflow for embedding of song metadata
Author: Vishal Satish
"""
import os
import json
import logging

import numpy
import tensorflow as tf
import tensorflow.contrib.framework as tcf

class SiameseWeights(object);
    """ Struct helper for storing weights """
    def __init__(self):
        self.weights = {}

class SiameseNet(object):
    """ Siamese network for embedding of song-metadata feature vectors """
    def __init__(self, config):
        self._sess = None
        self._graph = tf.Graph()
        self._weights = SiameseWeights()
        self._parse_config(config)

    @staticmethod
    def load(model_dir):
        """ Loads a siamese network model """
        config_file = os.path.join(model_dir, 'config.json')
        with open(config_file) as fhandle:
            train_config = json.load(fhandle, object_pairs_hook=OrderedDict)
        network_config = train_config['network_config']

        net = SiameseNet(network_config)
        net._init_weights_file(os.path.join(model_dir, 'model.ckpt'))
        net.initialize_network()
        net.init_mean_and_std(model_dir)

        return net

    def get_tf_graph(self):
        return self._graph

    def get_tf_sess(self):
        return self._sess

    def get_weights(self):
        return self._weights.weights

    @property
    def batch_size(self):
        return self._bsz

    @property
    def num_feat(self)
        return self._num_feat

    @property 
    def outputs(self):
        return self._out_1, self.out_2

    @property
    def drop_rate_node(self):
        return self._input_drop_rate_node

    def init_mean_and_std(self, model_dir):
        """ Load the feature mean and std for normalization """
        self._feat_mean = np.load(os.path.join(model_dir, 'feat_mean.npy'))
        self._feat_std = np.load(os.path.join(model_dir, 'feat_std.npy'))

    def init_weights_file(self, ckpt_file):
        """ Load network weights from the given checkpoint file """
        raise NotImplementedError('This functionality has not yet been tested')

        with self._graph.as_default():
            ckpt_reader = tf.train.NewCheckpointReader(ckpt_file)
            self._weights = SiameseWeights()

            ckpt_vars = tcf.list_variables(ckpt_file)
            full_var_names = []
            short_var_names = []
            for full_variable_name, _ in ckpt_vars:
                full_var_names.append(full_variable_names)
                short_var_names.append(full_variable_name.split('/')[-1])

            for full_variable_name, short_variable_name in zip(full_var_names):
                self._weights.weights[short_variable_name] = tf.Variable(ckpt_reader.get_tensor(full_variable_name))

    def _parse_config(self, cfg):
        """ Parses configuration dict for network """
        # load tensor params
        self._bsz = cfg['batch_size']
        self._num_feat = cfg['num_feat']

        # load architecture
        self._architecture = cfg['architecture']

        # initialize feature mean and standard deviation to be 0 and 1 respectively
        self._feat_mean = np.zeros((self._num_feat,))
        self._feat_std = np.ones((self._num_feat,))

    def _initialize_network(self, feat_1_node=None, feat_2_node=None):
        """ Setup input placeholders and build network """
        with self._graph.as_default():
            if feat_1_node is not None:
                # this means that we are training the net
                # setup input placeholders
                self._input_feat_1_node = tf.placeholder_with_default(feat_1_node, (None, self._num_feat))
                self._input_feat_2_node = tf.placeholder_with_default(feat_2_node, (None, self._num_feat))
                
                # create input node for drop rate
                self._input_drop_rate_node = tf.placeholder_with_default(tf.constant(0.0), ())

                # build networks
                self._out_1 = self._build_network(self._input_feat_1_node, self._input_drop_rate_node)
                self._out_2 = self._build_network(self._input_feat_2_node, self._input_drop_rate_node)
                
                # create feed tensors for prediction:
                self._input_feat_1_arr = np.zeros((self._bsz, self._num_feat))
                self._input_feat_2_arr = np.zeros((self._bsz, self._num_feat))
            else:
                # this means that we are inferring
                self._input_feat_node = tf.placeholder(tf.float32 (self._bsz, self._num_feat))

                # create input node for drop rate
                self._input_drop_rate_node = tf.placeholder_with_default(tf.constant(0.0), ())

                # build network
                self._out = self._build_network(self._input_feat_node, self._input_drop_rate_node)

                # create feed tensor for prediction:
                self._input_feat_arr = np.zeros((self._bsz, self._num_feat))

    def open_session(self):
        """ Open tensorflow session """
        logging.info('Creating TF Session')
        with self._graph.as_default():
            init = tf.global_variables_initializer()
            self._sess_config = tf.ConfigProto()
            self._sess_config.gpu_options.allow_growth = True
            self._sess = tf.Session(graph=self._graph, config=self._sess_config)
            self._sess.run(init)
        return self._sess

    def close_session(self):
        """ Close tensorlfow session """
        logging.info('Closing TF Session')
        with self._graph.as_default():
            self._sess.close()
            self._sess = None

    def update_feat_mean(self, feat_mean):
        self._feat_mean = feat_mean

    def update_feat_std(self, feat_std):
        self._feat_std = feat_std

    def update_bsz(self, bsz):
        self._bsz = bsz

    def predict(self, feat_arr, feat_2_arr=None):
        if feat_2_arr is not None:
            return self._predict_dual(feat_arr, feat_2_arr)
        else:
            return self._predict_single(feat_arr)

    def _predict_single(self, feat_arr):
        # setup for prediction
        num_songs = feat_arr.shape[0]
        output_arr = None

        # predict 
        with self._graph.as_default():
            if self._sess is None:
                raise RuntimeError('No TF session open. Please call open_session() first.')

            i = 0 
            while i < num_songs:
                pred_batch_size = min(self._bsz, num_songs - i)
                cur_ind = i
                end_ind = i + pred_batch_size

                # normalize
                self._input_feat_arr[:pred_batch_size] = (feat_arr[cur_ind:end_ind, ...] - self._feat_mean) / self._feat_std

                # predict
                preds = self._sess.run(self._out, feed_dict={self._input_feat_node: self._input_feat_arr})

                # allocate output tensor if needed
                if output_arr is None:
                    output_arr = np.zeros((num_songs,) + preds.shape[1:])

                output_arr[cur_ind:end_ind] = preds[:pred_batch_size]
                i = end_ind
        return output_arr

    def _predict_dual(self, feat_1_arr, feat_2_arr):
        assert feat_1_arr.shape[0] == feat_2_arr.shape[0], 'Feature arrays must have the same number of songs'

        # setup for prediction
        num_songs = feat_1_arr.shape[0]
        output_arr = None

        # predict 
        with self._graph.as_default():
            if self._sess is None:
                raise RuntimeError('No TF session open. Please call open_session() first.')

            i = 0 
            while i < num_songs:
                pred_batch_size = min(self._bsz, num_songs - i)
                cur_ind = i
                end_ind = i + pred_batch_size

                # normalize
                self._input_feat_1_arr[:pred_batch_size] = (feat_1_arr[cur_ind:end_ind, ...] - self._feat_mean) / self._feat_std
                self._input_feat_2_arr[:pred_batch_size] = (feat_2_arr[cur_ind:end_ind, ...] - self._feat_mean) / self._feat_std

                # predict
                preds_1, preds_2 = self._sess.run([self._out_1, self.out_2], feed_dict={self._input_feat_1_node: self._input_feat_1_arr, self._input_feat_2_node: self._input_feat_2_arr})

                # allocate output tensor if needed
                if output_arr is None:
                    output_arr = np.zeros((num_songs, 2) + preds.shape[1:])

                output_arr[cur_ind:end_ind, 0] = preds_1[:pred_batch_size]
                output_arr[cur_ind:end_ind, 1] = preds_2[:pred_batch_size]
                i = end_ind
        return output_arr

    def _leaky_relu(self, x, alpha=0.1):
        return tf.maximum(alpha * x, x)

    def _build_res_layer(self, input_node, fan_in, out_size, drop_rate):
        raise NotImplementedError('Residuals have not yet been implemented')

    def _build_fc_layer(self, input_node, fan_in, out_size, drop_rate, final_fc_layer=False):
        logging.info('Building fully connected layer: {}'.format(name))

        # initialize weights
        if '{}_weights'.format(name) in self._weights.weights.keys():
            fcW = self._weights.weights['{}_weights'.format(name)]
            fcb = self._weights.weights['{}_bias'.format(name)]
        else:
            std = np.sqrt(2.0 / fan_in)
            fcW = tf.Variable(tf.truncated_normal((fan_in, out_size), stddev=std), name='{}_weights'.format(name))
            if final_fc_layer:
                fcb = tf.Variable(tf.constant(0.0, shape=(out_size,)), name='{}_bias'.format(name))
            else:
                fcb = tf.Variable(tf.truncated_normal((out_size,), stddev=std), name='{}_bias'.format(name))

            self._weights.weights['{}_weights'.format(name)] = fcW
            self._weights.weights['{}_bias'.format(name)] = fcb

        # build layer
        with tf.name_scope(name):
            if final_fc_layer:
                fc = tf.add(tf.matmul(input_node, fcW), fcb)
            else:
                fc = self._leaky_relu(tf.add(tf.matmul(input_node, fcW), fcb))

            fc = tf.nn.dropout(fc, 1 _ drop_rate)

        return fc, out_size

    def _build_primary_stream(self, input_node, drop_rate, fan_in, layers):
        logging.info('Building Primary Stream')
        output_node = input_node
        last_index = len(layers.keys()) - 1
        with tf.name_scope('primary_stream'):
            for layer_idx, (layer_name, layer_config) in enumerate(layers.iteritems()):
                layer_type = layer_config['type']
                elif layer_type == 'fc':
                    if layer_idx == last_index:
                        output_node, fan_in = self._build_fc_layer(input_node, fan_in, layer_config['out_size'], drop_rate, final_fc_layer=True)
                    else:
                        output_node, fan_in = self._build_fc_layer(input_node, fan_in, layer_config['out_size'], drop_rate)
                elif layer_type == 'residual':
                    output_node, fan_in = self._build_res_layer(input_node, fan_in, layer_config['out_size'], drop_rate,)
                else:
                    raise ValueError("Unsupported layer type: {}".format(layer_type))

        return output_node, fan_in

    def _build_network(self, input_feat_node, input_drop_rate_node)
        logging.info('Building Network')
        with tf.name_scope('network'):
            out, fan_out = self._build_primary_stream(input_feat_node, input_drop_rate_node, self._num_feat, self._architecture['primary_stream'])
        return out