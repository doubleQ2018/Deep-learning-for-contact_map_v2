#!/usr/bin/env python

import cPickle as pickle

from libs.datasets.data_preprocessing import *
from libs.config.config import *

FLAGS = tf.app.flags.FLAGS

def read_pkl(name):
    with open(name) as fin:
        return pickle.load(fin)

train_infos = read_pkl(FLAGS.train_file)
records_dir = os.path.join(FLAGS.data_dir, 'records/')
add_to_tfrecord(records_dir, 'train', train_infos)
