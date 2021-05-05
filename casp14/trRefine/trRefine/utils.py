#!/usr/bin/env python3

import tensorflow as tf

class Json_param:
    def __init__(self, dict):
        vars(self).update(dict)

def reweight_seq(msa1hot, cutoff):
    with tf.variable_scope('reweight'):
        id_min = tf.cast(tf.shape(msa1hot)[1], tf.float32) * cutoff # msa1hot.shape[1] == n_res
        id_mtx = tf.tensordot(msa1hot, msa1hot, [[1,2], [1,2]])
        id_mask = id_mtx > id_min
        w = 1.0/tf.reduce_sum(tf.cast(id_mask, dtype=tf.float32),-1)
    return w

def msa2pssm(msa1hot, w):
    beff = tf.reduce_sum(w)
    f_i = tf.reduce_sum(w[:,None,None]*msa1hot, axis=0) / beff + 1e-9
    h_i = tf.reduce_sum( -f_i * tf.log(f_i+1e-9), axis=1)    
    return tf.concat([f_i, h_i[:,None]], axis=1)

# Peforms pixel wise spatial self-attention on an input matrix.
def pixelSelfAttention(x,
                       maxpool=1,
                       kq_factor=8,
                       v_factor=2,
                       reuse=False):
    
    # Get parameters
    bs, _, _, c = x.get_shape().as_list()
    bs = tf.shape(x)[0]
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    
    # Key
    f = tf.layers.conv2d(inputs=x,
                         filters=c//kq_factor,
                         kernel_size=1,
                         strides=1)
    
    if maxpool > 1: 
        f = tf.layers.max_pooling2d(f,
                                    pool_size=maxpool,
                                    strides=maxpool,
                                    padding='SAME')
    # Query
    g = tf.layers.conv2d(inputs=x,
                         filters=c//kq_factor,
                         kernel_size=1,
                         strides=1)
    
    # Value
    h = tf.layers.conv2d(inputs=x,
                         filters=c//v_factor,
                         kernel_size=1,
                         strides=1)
    
    if maxpool > 1: 
        h = tf.layers.max_pooling2d(h,
                                    pool_size=maxpool,
                                    strides=maxpool,
                                    padding='SAME')
    print(h)
        
    def hw_flatten(matrix):
        print ('check:', matrix.shape[0], matrix.shape[-1])
        return tf.reshape(matrix, shape=[tf.shape(matrix)[0], -1, matrix.shape[-1]])
        
    # Flattening and generating attention_map
    # Attention map: this should suppossed to be N by N whre N=hxc 
    attent_map = tf.matmul(hw_flatten(g),
                           hw_flatten(f),
                           transpose_b=True)
    beta = tf.nn.softmax(attent_map)
    print(beta)

    # Calculated infuluence 
    o = tf.matmul(beta,
                  hw_flatten(h))
    print ('check2', bs, height, width, c//v_factor)
    o = tf.reshape(o, shape=[bs, height, width, c//v_factor])
    o = tf.layers.conv2d(inputs=o,
                         filters=c,
                         kernel_size=1,
                         strides=1)
    
    # Initialize gamma to be 0.
    gamma = tf.Variable(0, dtype=tf.float32)
    
    # Adding back using gamma
    x = gamma * o + x
    return x

def f_arcsinh(X, cutoff=6.0, scaling=3.0):
    X_prime = tf.maximum(X, tf.zeros_like(X) + cutoff) - cutoff
    return tf.asinh(X_prime)/scaling
