#!/usr/bin/env python

import libs.nets.network as network
#import libs.nets.highway_network as network
import libs.datasets.data_preprocessing as data_preprocess
from libs.config.config import *
from libs.utils.acc_cal_v2 import topKaccuracy, evaluate, output_result

import tensorflow as tf
import numpy as np
import cPickle as pickle
import os
# using GPU numbered 0
os.environ["CUDA_VISIBLE_DEVICES"]='1'

def load_test_data():
    datafile = "data/pdb25-test-500.release.contactFeatures.pkl"
    f = open(datafile)
    data = pickle.load(f)
    f.close()
    return data

def test():
    # restore graph
    input_1d = tf.placeholder("float", shape=[None, None, 26], name="input_x1")
    input_2d = tf.placeholder("float", shape=[None, None, None, 5], name="input_x2")
    label = tf.placeholder("float", shape=None, name="input_y")
    output = network.build(input_1d, input_2d, label,
            FLAGS.filter_size_1d, FLAGS.filter_size_2d,
            FLAGS.block_num_1d, FLAGS.block_num_2d,
            regulation=True, batch_norm=True)
    prob = output['output_prob']

    # restore model
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
    #checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt-90000")
    print "Loading model from %s" %checkpoint_path
    restorer = tf.train.Saver()
    restorer.restore(sess, checkpoint_path)
    
    # prediction
    #predict single one
    #output_prob = sess.run(prob, feed_dict={input_1d: f_1d, input_2d: f_2d})
    data = load_test_data()
    input_acc = []
    output_acc = []
    for i in range(len(data)):
        d = data[i]
        name, seqLen, sequence_profile, pairwise_profile, true_contact = \
                data_preprocess.extract_single(d)
        #print "processing %d %s" %(i+1, name)
        sequence_profile = sequence_profile[np.newaxis, ...]
        #print sequence_profile.shape
        pairwise_profile = pairwise_profile[np.newaxis, ...]
        #print pairwise_profile.shape
        y_out = sess.run(prob, \
                feed_dict = {input_1d: sequence_profile, input_2d: pairwise_profile})
        #np.savetxt("results/"+name+".deepmat", y_out[0,:,:,1])
        #np.savetxt("contacts/"+name+".contacts", true_contact[0])
        input_acc.append(evaluate(pairwise_profile[0,:,:,0], true_contact))
        output_acc.append(evaluate(y_out[0,:,:,1], true_contact))

    print "Input result:"
    output_result(np.mean(np.array(input_acc), axis=0))
    print "\nOutput result:"
    output_result(np.mean(np.array(output_acc), axis=0))
    
if __name__ == "__main__":
    test()
