"""
Script for training Siamese Network to embed song metadata
Author: Vishal Satish
"""
import time
import logging
import yaml

from network import SiameseNet
from trainer import SiameseNetTrainer

# declare any enums and constants
CONFIG_FILE = 'cfg/training.yaml'

# setup logger
logging.getLogger().setLevel(logging.INFO)

# parse YAML config 
with open(CONFIG_FILE, 'r') as fstream:
    try:
        train_config = yaml.load(fstream)
    except yaml.YAMLError as e:
        logging.error(e)
        exit(0)
net_config = train_config['network_config']

def get_elapsed_time(time_in_seconds):
    """ Helper function to get elapsed time """
    if time_in_seconds < 60:
        return '%.1f seconds' % (time_in_seconds)
    elif time_in_seconds < 3600:
        return '%.1f minutes' % (time_in_seconds / 60)
    else:
        return '%.1f hours' % (time_in_seconds / 3600)

# train
start_time = time.time()
network = SiameseNet(net_config)
trainer = SiameseNetTrainer(network, train_config)
trainer.train()
logging.info('Total training time: {}'.format(str(get_elapsed_time(time.time() - start_time))))
