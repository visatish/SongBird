"""
Trains a Siamese Network in Tensorflow to embed song metadata
Author: Vishal Satish
"""
import logging
import cPickle as pkl
import os
import signal
import threading
import time

import numpy as np
import tensorflow as tf

class SiameseNetTrainer(object):
    """ Trains Siamese Net with Tensorflow backend """
    def __init__(self, net, config):
        self._net = net
        self._cfg = config
        self._tensorboard_has_launched = False

    def _create_loss(self):
        """ 
        Builds Loss:
        || lambda * iou - || x1 - x2 ||^2 ||^2
        """
        with tf.name_scope('loss_function'):
            loss = tf.reduce_mean(tf.square(tf.subtract(tf.multiply(self._loss_reg_coeff, self._train_labels), tf.square(tf.norm(tf.subtract(self._out_1, self._out_2), axis=1)))))
        return loss

    def _create_optimizer(self, loss, g_step, var_list, lr):
        """
        Builds Optimizer
        """
        if self._cfg['optimizer'] == 'momentum':
            optimizer = tf.train.MomentumOptimizer(lr, self._momentum)
            return optimizer.minimize(loss, global_step=g_step, var_list=var_list), optimizer
        elif self._cfg['optimizer'] == 'adam':
            optimizer = tf.train.AdamOptimizer(lr)
            return optimizer.minimize(loss, global_step=g_step, var_list=var_list), optimizer
        elif self._cfg['optimizer'] == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(lr)
            return optimizer.minimize(loss, global_step=g_step, var_list=var_list), optimizer
        else:
            raise ValueError('Optimizer {} is not supported'.format(self._cfg['optimizer']))

    def _check_dead_queue(self):
        """ Checks to see if the queue is dead and if so closes the Tensorflow session and cleans up the variables """
        if self._dead_event.is_set():
            # close sess
            self._sess.close()

            # cleanup
            for layer_weights in self._weights.values():
                del layer_weights
            del self._saver
            del self._sess

    def _launch_tensorboard(self):
        """ Launches Tensorboard """
        logging.info('Launching Tensorboard, Please navigate to {} in your favorite web browser to view summaries'.format(self._tensorboard_port))
        os.system('tensorboard --logdir={} &>/dev/null &'.format(self._summary_dir))

    def _close_tensorboard(self):
        """ Closes Tensorboard """
        logging.info('Closing Tensorboard...')
        tensorboard_pid = os.popen('pgrep tensorboard').read()
        os.system('kill {}'.format(tensorboard_pid))

    def train(self):
        """ Wrapper for _train that sets default graph to self._net's graph """
        with self._net.get_tf_graph().as_default():
            self._train()

    def _train(self):
        """ Perform Optimization """
        train_start_time = time.time()

        # run setup
        self._setup()

        # build training networks
        self._net.initialize_network(self._input_feat_1, self.input_feat_2)
        self._out_1, self.out_2 = self._net.outputs
        self._drop_rate_node = self._net.drop_rate_node

        # create Tensorflow Saver
        self._saver = tf.train.Saver()

        # form loss
        with tf.name_scope('loss')
            # part 1: loss func
            loss = self._create_loss()
            # part 2: weight regularization
            with tf.name_scope('weight regularization')
                layer_weights = self._weights.values()
                regularizer = tf.nn.l2_loss(layer_weights[0])
                for w in layer_weights[1:]:
                    regularizer = tf.add(regularizer, tf.nn.l2_loss(w))
            loss = tf.add(loss, tf.multiply(self._l2_weight_regularizer, regularizer))  

        # create global step Tensorflow Variable 
        g_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(self._base_lr, tf.multiply(g_step, self._train_bsz), self._decay_step, self._decay_rate, staircase=True)

        # get Tensorflow Variables to train(weights & biases)
        train_vars = self._weights.values()

        # create optimizer
        with tf.name_scope('optimizer'):
            optimization_op, optimizer = self._create_optimizer(loss, g_step, train_vars, learning_rate)

        # define handler func for data prefetch thread
        def handler(signum, frame):
            logging.info('Caught CTRL+C, exiting...')
            self._term_event.set()

            ### Forcefully Exit ####
            # TODO: remove this and figure out why queue thread does not properly exit
            logging.info('Forcefully Exiting Optimization')
            self._forceful_exit = True

            # forcefully kill the session to terminate any current graph 
            # ops that are stalling because the enqueue op has ended
            self._sess.close()

            # close tensorboard
            self._close_tensorboard()

            # pause and wait for queue thread to exit before continuing
            logging.info('Waiting for Queue Thread to Exit...')
            while not self.queue_thread_exited:
                pass

            logging.info('Cleaning and Preparing to Exit Optimization...')

            # cleanup
            for layer_weights in self._weights.values():
                del layer_weights
            del self.saver
            del self.sess

            # exit
            logging.info('Exiting Optimization')

            # forcefully exit the script
            exit(0)

        signal.signal(signal.SIGINT, handler)

        # add the Tensorflow graph to the summary writer
        self._summary_writer.add_graph(self._net.get_tf_graph())

        # begin optimization loop
        try:
            self._queue_thread = threading.Thread(target=self._load_and_enqueue)
            self._queue_thread.start()

            # initialize all Tensorflow global variables
            global_init = tf.global_variables_initializer()
            self._sess.run(init)

            logging.info('Beginning Optimization...')

            # loop through training steps
            training_range = xrange(int(self._num_epochs * self._num_train) // self._train_bsz)
            for step in training_range:
                # check for dead prefetch queue
                self._check_dead_queue()

                # fprop + bprop
                _, l, lr, preds_1, preds_2, batch_labels = self._sess.run([optimizer, loss, lr, self._out_1, self._out_2, self._train_labels_node], feed_dict={self._drop_rate_node: self._drop_rate})

                # log 
                if step % self._log_frequency == 0:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    logging.info('Step {} (epoch {:.2f}), {:.1f} s'.format(step, float(step) * self._train_bsz / self._num_train, 1000 * elapsed_time / self.log_frequency))
                    logging.info('Minibatch loss: {:.3f}, learning rate: {:.6f}'.format(l, lr))

                # update Tensorflow summaries
                self._summary_writer.add_summary(self._sess.run(self._merged_train_summary_op, feed_dict={self._minibatch_loss_placeholder: l, self._minibatch_lr_placeholder: lr}))

                # evaluate validation error
                if step % self._eval_frequency == 0:
                    val_error = self._error_rate_in_batches()
                    self._summary_writer.add_summary(self._sess.run(self._merged_val_summary_op, feed_dict={self._val_error_placeholder: val_error}))
                    logging.info('Validation error: {:.3f}'.format(val_error))

                # save the Tensorflow checkpoint file
                if step % self._save_frequency == 0 and step > 0:
                    self._saver.save(self._sess, self._experiment_path_gen('model_{:05d}.ckpt'.format(step)))
                    self._saver.save(self._sess, self._experiment_path_gen('model.ckpt'))

                # launch Tensorboard only after the first iteration
                if not self._tensorboard_has_launched:
                    self._tensorboard_has_launched = True
                    self._launch_tensorboard()

            # get final validation error
            val_error = self._get_val_error()
            logging.info('Final validation error: {:.1f}'.format(val_error))

            # save the final Tensorflow checkpoint
            self._saver.save(self._sess, self._experiment_path_gen('model.ckpt'))
        except Exception as e:
            self._term_event.set()
            if not self._forceful_exit:
                self._sess.close()
                for layer_weights in self._weights.values():
                    del layer_weights
                del self._saver
                del self._sess
            raise e

        # check for dead prefetch queue
        self._check_dead_queue()

        # set termination event
        self._term_event.set()

        # close Tensorboard process
        self._close_tensorboard()

        # TODO: remove this and figure out why prefetch queue thread does not properly exit
        self._sess.close()

        # pause and wait for prefetch queue thread to exit before continuing
        logging.info('Waiting for Prefetch Queue Thread to Exit...')
        while not self._queue_thread_exited:
            pass

        logging.info('Cleaning up and Preparing to Exit Optimization...')
        self._sess.close()

        # cleanup
        for layer_weights in self._weights.values():
            del layer_weights
        del self._saver
        del self._sess

        logging.info('Exiting Optimization')

    def _save_indices(self, train_idcs, val_idcs, train_fname, val_fname):
        """ Save training and validation indices """
        np.savez_compressed(train_fname, train_idcs)
        np.savez_compressed(val_fname, val_idcs)

    def _compute_indices(self):
        """ Computes training and validation indices """
        logging.info('Computing training and validation indices...')

        train_idcs_fname = 'train_indices.pkl'
        val_idcs_fname = 'val_indices.pkl'

        self._num_total = int(self._total_pct * self._num_datapoints)
        self._num_train = int(self._train_pct * self._num_total)

        all_indices = np.arange(self._num_datapoints)
        np.random.shuffle(all_indices)
        all_indices = all_indices[:self._num_total]
        train_idcs = all_indices[:self._num_train]
        val_idcs = all_indices[self._num_train:]

        self._save_indices(train_idcs, val_idcs, train_idcs_fname, val_idcs_fname)

        return train_idcs, val_idcs

    def _compute_data_metrics(self):
        """ Computes feature mean and std """
        logging.info('Computing data metrics...')
        feat_mean_fname = self._experiment_path_gen('feat_mean.npy')
        feat_std_fname = self._experiment_path_gen('feat_std.npy')

        # sample indices
        assert self._metric_sample_size <= self._num_train, 'Metric sample size must be <= size of training split'
        feat_ind = np.random.choice(self._train_idcs, size=self._metric_sample_size)

        # allocate feature tensor
        feat = np.zeros((self._metric_sample_size, self._num_feat))

        # load features
        feat_songs = self._dataset[feat_ind.tolist()]
        for i in range(self._metric_sample_size):
            feat[i, ...] = feat_songs[i].features

        # calculate mean and std
        self._feat_mean = np.mean(feat, axis=0)
        self._feat_std = np.sqrt(np.mean(np.square(self.feat - self._feat_mean), axis=0))

        np.save(feat_mean_fname, self._feat_mean)
        np.save(feat_std_fname, self._feat_std)

        self._net.update_feat_mean(self._feat_mean)
        self._net.update_feat_std(self._feat_std)

    def _setup_summaries(self):
        logging.info('Setting up Tensorflow summaries...')
        """ Sets up placeholders for summary values and creates summary writer """
        self._val_error_placeholder = tf.placeholders(tf.float32, ())
        self._minibatch_loss_placeholder = tf.placeholder(tf.float32, ())
        self._minibatch_lr_placeholder = tf.placeholder(tf.float32, ())

        # create summary scalars with tags to group summaries that are logged at the same interval together
        tf.summary_scalar('val_error', self._val_error_placeholder, collections=['eval_frequency'])
        tf.summary_scalar('minibatch_loss', self._minibatch_loss_placeholder, collections='log_frequency')
        tf.summary_scalar('minibatch_lr', self._minibatch_lr_placeholder, collections='log_frequency')

        # create merged summary ops for convenience
        self._merged_train_summary_op = tf.summary.merge_all('log_frequency')
        self._merged_val_summary_op = tf.summary.merge_all('eval_frequency')

        # create a Tensorflow summary writer
        self._summary_writer = tf.summary.FileWriter(self._summary_dir)

        # initialize the global variables again now that we have added new ones
        with self._sess.as_default():
            init = tf.global_variables_initializer()
            self._sess.run(init)

    def _setup_data_pipeline(self):
        """ Setup Tensorflow data pipeline for reading in data from dataset and forwarding it to model """
        # setup placeholders
        logging.info('Building data pipeline...')
        with tf.name_scope('input_feature_1_queue_batch'):
            # feature one batch for queue
            self._input_feat_1_queue_batch = tf.placeholder(tf.float32, (self._train_bsz, self._num_feat))
        with tf.name_scope('input_feature_2_queue_batch'):
            # feature two batch for queue
            self._input_feat_2_queue_batch = tf.placeholder(tf.float32, (self._train_bsz, self._num_feat))
        with tf.name_scope('input_IOU_labels_queue_batch'):
            self._input_labels_queue_batch = tf.placeholder(tf.float32, (self._train_bsz,))

        # create data prefetch queue
        with tf.name_scope('prefetch_queue'):
            self._prefetch_queue = tf.FIFOQueue(self._queue_cap, (tf.float32, tf.float32, tf.float32), shapes=[(self._train_bsz, self._num_feat), (self._train_bsz, self._num_feat), (self._train_bsz,)])
            self._enqueue_op = self._prefetch_queue.enqueue([self._input_feat_1_queue_batch, self._input_feat_2_queue_batch, self._input_labels_queue_batch])
            self._input_feat_1, self.input_feat_2, self._train_labels = self._prefetch_queue.dequeue()

            # get network weights
            self._weights = self._net.get_weights()

            # open a tf Session for the network and store it as self._sess
            self._sess = sels._net.open_session()

            # set up term event/dead event
            self._term_event = threading.Event()
            self._term_event.clear()
            self._dead_event = threading.Event()
            self._dead_event.clear()

    def _read_training_params(self):
        """ Read training params from config file """
        logging.info('Reading training parameters...')

        self._total_pct = self._cfg['total_pct']
        self._train_pct = self._cfg['train_pct']

        self._train_bsz = self._cfg['train_bsz']
        self._val_bsz = self._cfg['val_bsz']
        # update SiameseNetwork's bsz to self._val_bsz
        self._net.update_bsz(self._val_bsz)

        self._num_epochs = self._cfg['num_epochs']
        self._eval_frequency = self._cfg['eval_frequency']
        self._save_frequency = self._cfg['save_frequency']
        self._log_frequency = self._cfg['log_frequency']

        self._queue_cap = self._cfg['queue_capacity']
        self._queue_sleep = self._cfg['queue_sleep']

        self._l2_weight_regularizer = self._cfg['l2_weight_regularizer']
        self._base_lr = self._cfg['base_lr']
        self._decay_step_multiplier = self._cfg['decay_step_multiplier']
        self._decay_rate = self._cfg['decay_rate']
        self._momentum = self._cfg['momentum']
        self._drop_rate = self._cfg['drop_rate']

        self._metric_sample_size = self._cfg['metric_sample_size']

    def _read_data_params(self):
        """ Read data params from config file """
        logging.info('Reading data parameters...')

        self._data_dir = self._cfg['dataset_dir']
        self._num_datapoints = self._cfg['num_datapoints']
        self._num_feat = self._cfg['num_feat']

    def _gen_experiment_id(self, num_chars=10):
        """ Generate a random string of characters of length num_chars """
        chars = 'abcdefghijklmnopqrstuvwxyz'
        inds = np.random.randint(0, len(chars), size=num_chars)
        return ''.join([chars[i] for i in inds])

    def _setup_output_dirs(self, output_dir):
        """ Setup output directories """
        logging.info('Setting up output directories...')

        # setup experiment_dir
        experiment_id = self._gen_experiment_id()
        experiment_dir = os.path.join(output_dir, 'model_{}'.format(experiment_id))
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)

        # setup summary_dir    
        summary_dir = os.path.join(experiment_dir, 'summaries')
        if not os.path.exists(summary_dir):
            os.mkdir(summary_dir)
        else:
            # if the summary directory already exists, clean it out by deleting all files in it,
            # we don't want tensorboard to get confused with old logs while debugging with the same directory
            old_files = os.listdir(summary_dir)
            for file in old_files:
                os.remove(os.path.join(summary_dir, file))

        logging.info('Saving model to {}'.format(experiment_dir))

        return experiment_dir, summary_dir

    def _copy_config(self, experiment_dir, cfg):
        """ Copy entire training configuration dict to JSON file in experiment_dir. Also copy
        training script to experiment_dir. """
        logging.info('Copying training files...')

        # copy entire training configuration dict
        out_config_filename = os.path.join(experiment_dir, 'config.json')
        tempOrderedDict = collections.OrderedDict()
        for key in cfg.keys():
            tempOrderedDict[key] = cfg[key]
        with open(out_config_filename, 'w') as outfile:
            json.dump(tempOrderedDict, outfile)
        
        # copy training script
        this_filename = sys.argv[0]
        out_train_filename = os.path.join(experiment_dir, 'training_script.py')
        shutil.copyfile(this_filename, out_train_filename)

    def _get_decay_step(self, train_pct, num_total_datapoints, decay_step_multiplier):
        num_steps_in_epoch = int(train_pct * num_total_datapoints / self._train_bsz)
        return decay_step_multiplier * num_steps_in_epoch
    
    def _open_song_dataset(self):
        """ Opens a SongDataset """
        logging.info('Opening dataset...')

        return SongDataset(self._data_dir)

    def _setup(self):
        """ Set up for training """

        # set random seed for deterministic execution if in debug mode
        self._debug = self._cfg['debug'] 
        if self._debug:
            np.random.seed(RANDOM_SEED)
            random.seed(RANDOM_SEED)

        # initialize thread exit booleans
        self._queue_thread_exited = False
        self._forceful_exit = False

        # setup output directories
        output_dir = self.cfg['output_dir']
        self._experiment_dir, self._summary_dir = self._setup_output_dirs(output_dir)

        # create python lambda function to help create file paths to experiment_dir
        self._experiment_path_gen = lambda fname: os.path.join(self._experiment_dir, fname)

        # copy config file
        self._copy_config(self._experiment_dir, self._cfg)

        # read training parameters from config file
        self._read_training_params()

        # read data parameters from config file
        self._read_data_params()

        # initialize SongDataset
        self._dataset = self._open_song_dataset()

        # compute training and validation indices
        self._train_idcs, self._val_idcs = self._compute_indices()

        steps_per_epoch = self._num_total * self._train_pct / self._train_bsz
        # if self._eval_frequency == -1, change it to reflect a single epoch
        if self._eval_frequency == -1:
            self._eval_frequency = steps_per_epoch

        # if self.save_frequency == -1, change it to reflect a single epoch
        if self._save_frequency == -1:
            self._save_frequency = steps_per_epoch

        # calculate learning rate decay step
        self._decay_step = self._get_decay_step(self._train_pct, self._num_total, self._decay_step_multiplier)

        # compute data metrics
        self._compute_data_metrics()

        # setup tensorflow data pipeline
        self._setup_data_pipeline()

        # setup Tensorflow summaries
        self._setup_summaries()

    def _load_and_enqueue(self):
        """ Loads and enqueues a batch of features and labels """

        # allocate intermediate buffers
        input_feat_1_batch = np.zeros((self._train_bsz, self._num_feat))
        input_feat_2_batch = np.zeros((self._train_bsz, self._num_feat))
        input_labels_batch = np.zeros((self._train_bsz,))

        # loop while self._term_event is not set
        while not self._term_event.is_set():
            # sample indices
            feat_1_ind = np.random.choice(self._train_idcs, size=self._train_bsz)
            feat_2_ind = np.random.choice(self._train_idcs, size=self._train_bsz)

            # load from dataset and into intermediate buffers
            feat_1_songs = self._dataset[feat_1_ind.tolist()]
            feat_2_songs = self._dataset[feat_2_ind.tolist()]
            pairwise_ious = self._dataset.get_IOU(feat_1_songs, feat_2_songs)
            for i in range(self._train_bsz):
                input_feat_1_batch[i, ...] = feat_1_songs[i].features
                input_feat_2_batch[i, ...] = feat_2_songs[i].features
                input_labels_batch[i] = pairwise_ious[i]

            # normalize features
            input_feat_1_batch = (input_feat_1_batch - self._feat_mean) / self._feat_std
            input_feat_2_batch = (input_feat_2_batch - self._feat_mean) / self._feat_std  

            # enqueue buffers
            if not self._term_event.is_set():
                try:
                    self._sess.run(self._enqueue_op, feed_dict={self._input_feat_1_queue_batch: input_feat_1_batch, self._input_feat_2_queue_batch: input_feat_2_batch, self._input_labels_queue_batch: input_labels_batch})
                except:
                    pass
        self._dead_event.set()
        logging.info('Queue Thread Exiting')
        self._queue_thread_exited = True

    def _get_val_error(self):
        """ Computes validation error """
         
         # sample indices
         num_samples = self._val_idcs.shape[0]
         feat_1_idcs = np.random.choice(self._val_idcs, size=num_samples)
         feat_2_idcs = np.random.choice(self._val_idcs, size=num_samples)

         # allocate tensors
         feat_1 = np.zeros((num_samples, self._num_feat))
         feat_2 = np.zeros((num_samples, self._num_feat))
         labels = np.zeros((num_samples,))

         # load from dataset
         feat_1_songs = self._dataset[feat_1_idcs.tolist()]
         feat_2_songs = self._dataset[feat_2_idcs.tolist()]
         pairwise_ious = self._dataset.get_IOU(feat_1_songs, feat_2_songs)
         for i in range(num_samples):
            feat_1[i, ...] = feat_1_songs[i].features
            feat_2[i, ...] = feat_2_songs[i].features
            labels[i] = pairwise_ious[i]

        # predict
        preds = self._net.predict(feat_1, feat_2)

        # calculate error as || iou - || x1 - x2 ||^2 ||^2
        return np.mean(np.square(labels - np.square(np.linalg.norm(feat_1 - feat_2, axis=1))))