#!/usr/bin/env python

import tensorflow as tf
import glob
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType

def get_dataset(split_name, dataset_dir, file_pattern=None):
    if file_pattern is None:
        file_pattern = split_name + '*.tfrecord' 
    tfrecords = glob.glob(dataset_dir + '/records/' + file_pattern)
    name,seqLen, seq_feature, pair_feature, label = read_tfrecord(tfrecords)

    return name, seqLen, seq_feature, pair_feature, label 


def read_tfrecord(tfrecords_filename):
    if not isinstance(tfrecords_filename, list):
        tfrecords_filename = [tfrecords_filename]
    filename_queue = tf.train.string_input_producer(
        tfrecords_filename, num_epochs=100)

    options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
    reader = tf.TFRecordReader(options=options)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
      'name': tf.FixedLenFeature([], tf.string),
      'seqLen': tf.FixedLenFeature([], tf.int64),
      'seq_feature': tf.FixedLenFeature([], tf.string),
      'pair_feature': tf.FixedLenFeature([], tf.string),
      'label_matrix': tf.FixedLenFeature([], tf.string),
      })
    name = features['name']
    seqLen = tf.cast(features['seqLen'], tf.int32)
    seq_feature = tf.decode_raw(features['seq_feature'], tf.float32)
    seq_feature = tf.reshape(seq_feature, [seqLen, -1])             # reshape seq feature to shape = (L, feature_maps)
    pair_feature = tf.decode_raw(features['pair_feature'], tf.float32)
    pair_feature = tf.reshape(pair_feature, [seqLen, seqLen, -1])   # reshape pair feature to shape = (L, L, feature_maps)
    label = tf.decode_raw(features['label_matrix'], tf.uint8)       
    label = tf.reshape(label, [seqLen, seqLen, 1])                  # reshape label to shape = (L, L, 1)

    return name, seqLen, seq_feature, pair_feature, label

def test():
    dataset_dir = "data/"
    split_name = "train"
    name, seqLen, seq_feature, pair_feature, label = get_dataset(split_name, dataset_dir)

    init = tf.initialize_local_variables()
    sess = tf.Session()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    name, seqLen, seq, pair, label = sess.run([name,seqLen, seq_feature, pair_feature, label])
    print name
    print seqLen
    print seq.shape
    print pair.shape
    for l in label:
        print ''.join([str(i) for i in l])

#test()
