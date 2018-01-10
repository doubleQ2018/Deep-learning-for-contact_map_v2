#!/usr/bin/env python

import tensorflow as tf
import numpy as np

def weight_variable(shape, regularizer, name="W"):
    if regularizer == None:
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name)
    else:
        return tf.get_variable(name, shape, 
                initializer=tf.random_normal_initializer(), regularizer=regularizer)


def bias_variable(shape, name="b"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)

### Incoming shape (batch_size, L(seqLen), feature_num)
### Output[:, i, j, :] = incoming[:. i, :] + incoming[:, j, :] + incoming[:, (i+j)/2, :]
def seq2pairwise(incoming):
    L = tf.shape(incoming)[1]
    #save the indexes of each position
    v = tf.range(0, L, 1)
    i, j = tf.meshgrid(v, v)
    m = (i+j)/2
    #switch batch dim with L dim to put L at first
    incoming2 = tf.transpose(incoming, perm=[1, 0, 2])
    #full matrix i with element in incomming2 indexed i[i][j]
    out1 = tf.nn.embedding_lookup(incoming2, i)
    out2 = tf.nn.embedding_lookup(incoming2, j)
    out3 = tf.nn.embedding_lookup(incoming2, m)
    #concatante final feature dim together
    out = tf.concat([out1, out2, out3], axis=3)
    #return to original dims
    output = tf.transpose(out, perm=[2, 0, 1, 3])
    return output

def build_block_1d(incoming, out_channels, filter_size, 
        regularizer, batch_norm=False, scope=None, name="ResidualBlock_1d"):

    net = incoming
    in_channels = incoming.get_shape().as_list()[-1]
    ident = net
    with tf.variable_scope(scope, default_name = name, values=[incoming]) as scope:
        # 1st conv layer in residual block
        W1 = weight_variable([filter_size, in_channels, out_channels], regularizer, name="W1")
        #variable_summaries(W1)
        b1 = bias_variable([out_channels], name="b1")
        #variable_summaries(b1)
        net = tf.nn.conv1d(net, W1, stride=1, padding='SAME') + b1
        ### Add batch nomalization
        if batch_norm:
            net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu(net)
        # 2nd conv layer in residual block
        W2 = weight_variable([filter_size, out_channels, out_channels], regularizer, name="W2")
        #variable_summaries(W2)
        b2 = bias_variable([out_channels], name="b2")
        #variable_summaries(b2)
        net = tf.nn.conv1d(net, W2, stride=1, padding='SAME') + b2
        ### Add batch nomalization
        if batch_norm:
            net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu(net)  
        if in_channels != out_channels:
            ch = (out_channels - in_channels)//2
            remain = out_channels-in_channels-ch
            ident = tf.pad(ident, [[0, 0], [0, 0], [ch, remain]])
            in_channels = out_channels
        # Add the original featrues to result, identify
        net = net + ident
    return net

def build_block_2d(incoming, out_channels, filter_size, 
        regularizer, batch_norm=False, scope=None, name="ResidualBlock_2d"):

    net = incoming
    in_channels = incoming.get_shape().as_list()[-1]
    ident = net
    with tf.variable_scope(scope, default_name = name, values=[incoming]) as scope:
        # 1st conv layer in residual block
        W1 = weight_variable([filter_size, filter_size, in_channels, out_channels], regularizer, name="W1")
        #variable_summaries(W1)
        b1 = bias_variable([out_channels], name="b1")
        #variable_summaries(b1)
        net = tf.nn.conv2d(net, W1, strides=[1,1,1,1], padding='SAME') + b1
        ### Add batch nomalization
        if batch_norm:
            net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu(net)
        ### 2nd conv layer in residual block
        W2 = weight_variable([filter_size, filter_size, out_channels, out_channels], regularizer, name="W2")
        #variable_summaries(W2)
        b2 = bias_variable([out_channels], name="b2")
        #variable_summaries(b2)
        net = tf.nn.conv2d(net, W2, strides=[1,1,1,1], padding='SAME') + b2
        ### Add batch nomalization
        if batch_norm:
            net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu(net)  
        if in_channels != out_channels:
            ch = (out_channels - in_channels)//2
            remain = out_channels-in_channels-ch
            ident = tf.pad(ident, [[0, 0], [0, 0], [0, 0], [ch, remain]])
            in_channels = out_channels
        ### Add the original featrues to result
        net = net + ident
    return net

def one_hot(contact_map):
    # change the shape to (L, L, 2) 
    tmp = np.where(contact_map > 0, 0, 1)
    true_contact = np.stack((tmp, contact_map), axis=-1)
    return true_contact.astype(np.float32)

def build_loss(output_prob, y, weight=None):
    y = tf.py_func(one_hot, [y], tf.float32)
    los = -tf.reduce_mean(tf.multiply(tf.log(tf.clip_by_value(output_prob,1e-10,1.0)), y))
    return los

def build(input_1d, input_2d, label, 
        filter_size_1d=17, filter_size_2d=3, block_num_1d=0, block_num_2d=10,
        regulation=True, batch_norm=True):
    
    regularizer = None
    if regulation:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    net = input_1d

    channel_step = 2
    ######## 1d Residual Network ##########
    out_channels = net.get_shape().as_list()[-1]
    for i in xrange(block_num_1d):    #1D-residual blocks building
        out_channels += channel_step
        net = build_block_1d(net, out_channels, filter_size_1d, 
                regularizer, batch_norm=batch_norm, name="ResidualBlock_1D_"+str(i))
            
    #######################################
    
    # Conversion of sequential to pairwise feature
    with tf.name_scope('1d_to_2d'):
        net = seq2pairwise(net) 

    # Merge coevolution info(pairwise potential) and above feature
    if block_num_1d == 0:
        net = input_2d
    else:
        net = tf.concat([net, input_2d], axis=3)
    out_channels = net.get_shape().as_list()[-1]
    
    ######## 2d Residual Network ##########
    for i in xrange(block_num_2d):    #2D-residual blocks building
        out_channels += channel_step
        net = build_block_2d(net, out_channels, filter_size_2d, 
                regularizer, batch_norm=batch_norm, name="ResidualBlock_2D_"+str(i))
    #######################################

    # softmax channels of each pair into a score
    with tf.variable_scope('softmax_layer', values=[net]) as scpoe:
        W_out = weight_variable([1, 1, out_channels, 2], regularizer, 'W')
        b_out = bias_variable([2], 'b')
        output_prob = tf.nn.softmax(tf.nn.conv2d(net, W_out, strides=[1,1,1,1], padding='SAME') + b_out)
    
    with tf.name_scope('loss_function'):
        loss = build_loss(output_prob, label)
        if regulation:
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            loss += reg_term
        tf.summary.scalar('loss', loss)
    output = {}
    output['output_prob'] = output_prob
    output['loss'] = loss

    return output


